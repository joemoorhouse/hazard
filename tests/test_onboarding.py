import logging
import os
from resource import RUSAGE_SELF, getrusage
import sys
import threading
from pathlib import Path, PurePosixPath

import fsspec.implementations.local as local  # type: ignore
import numpy as np
import psutil
import pytest
import s3fs
import tempfile as temp
import zarr
from dask.distributed import Client, LocalCluster
from hazard import get_hazards_onboarding
from hazard.docs_store import DocStore
from hazard.indicator_model import IndicatorModel
from hazard.models.days_tas_above import DaysTasAboveIndicator
from hazard.models.degree_days import HeatingCoolingDegreeDays
from hazard.models.multi_year_average import (
    MULTI_MODEL_ID,
    RunConfig,
    ThresholdBasedAverageIndicator,
    ThresholdBatchItem,
)
from hazard.models.prechunker import NexGddpCmip6CachedSource, download_paths
from hazard.models.wet_bulb_globe_temp import DaysWBGTAboveIndicator
from hazard.onboard.general_onboarding import initialize_class, onboard_hazards_to
from hazard.sources.osc_zarr import OscZarr
from hazard.utilities import s3_utilities, zarr_utilities


@pytest.fixture
def s3_credentials():
    zarr_utilities.set_credential_env_variables()
    yield "s3_credentials"


@pytest.fixture
def log_to_stdout():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    yield "log_to_stdout"


@pytest.fixture
def log_to_file(request, test_output_dir):
    """Also write the log records of all loggers to test_output/logs/<test name>.log, overwritten each run.

    This is in addition to any other logging (e.g. pytest's live logging), whether the test is
    run or debugged.
    """
    path = os.path.join(test_output_dir, "logs", f"{request.node.name}.log")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handler = logging.FileHandler(path, mode="w")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
        )
    )
    root = logging.getLogger()
    previous_level = root.level
    root.addHandler(handler)
    # ensure INFO records reach the handler, whatever pytest's logging configuration
    root.setLevel(min(previous_level, logging.INFO))
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()
        root.setLevel(previous_level)


@pytest.fixture
def log_memory():
    """Log the memory use of the test process, and of the system, every 30 s and at the end of the test.

    With a dask LocalCluster(processes=False) the workers are threads of the test process,
    so its memory is that of the whole calculation. List after log_to_file among the test's
    arguments, so that the final record also reaches the log file.
    """
    interval = 30.0
    process = psutil.Process()
    peak_rss = 0
    stop = threading.Event()

    def log_usage(prefix: str):
        nonlocal peak_rss
        rss = process.memory_info().rss
        peak_rss = max(peak_rss, rss)
        system = psutil.virtual_memory()
        swap = psutil.swap_memory()
        logging.info(
            f"{prefix}process RSS {rss / 1e9:.1f} GB (peak {peak_rss / 1e9:.1f} GB); "
            f"system available {system.available / 1e9:.1f} of {system.total / 1e9:.1f} GB, "
            f"swap used {swap.used / 1e9:.1f} GB"
        )

    def monitor():
        while not stop.wait(interval):
            log_usage("Memory: ")

    log_usage("Memory at start: ")
    thread = threading.Thread(target=monitor, name="log_memory", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()
        # peak sampled every interval may miss short spikes: ru_maxrss (kB on Linux) does not
        max_rss = getrusage(RUSAGE_SELF).ru_maxrss * 1e3
        log_usage("Memory at end: ")
        logging.info(f"Memory: maximum process RSS {max_rss / 1e9:.1f} GB")


@pytest.fixture
def test_output_dir():
    """Provides directory for (for example) testing (file-based) storage of datasets."""
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_output")
    yield output_dir


@pytest.fixture
def onboarding_output_dir():
    """Provides directory for onboarding data tests, organized in the same way for all hazard onboards.

    It checks whether you have it already in your temporary files and creates it for you.
    """
    fs = local.LocalFileSystem()
    temp_dir = os.path.join(temp.gettempdir(), "onboarding_raw_data")

    if not fs.exists(temp_dir):
        temp_dir = temp.TemporaryDirectory()
        new_temp_dir_path = os.path.join(
            os.path.dirname(temp_dir.name), "onboarding_raw_data"
        )
        os.rename(temp_dir.name, new_temp_dir_path)
        temp_dir.name = new_temp_dir_path
        os.makedirs(
            os.path.join(new_temp_dir_path, "hazard", "hazard.zarr"), exist_ok=True
        )
        os.makedirs(os.path.join(new_temp_dir_path, "downloads"), exist_ok=True)
        temp_dir._finalizer.detach()
        temp_dir = temp_dir.name

    yield temp_dir


@pytest.mark.manual
def test_onboard_named_hazard_local(test_output_dir):
    """Used to onboard into a local directory a named hazard."""
    hazard_to_onboard = "JRCRiverFlood"  # "DavydzenkaEtAlLandSubsidence" # "JRCRiverFlood" # "WRIAqueductWaterRisk"
    maps_only = False  # True to skip prepare/onboard and just regenerate maps
    # from array data already in test_output_dir (e.g. while iterating on map code).

    store = zarr.DirectoryStore(
        os.path.join(test_output_dir, "hazard-indicators", "hazard.zarr")
    )
    target = OscZarr(store=store)

    if maps_only:
        hazard_map = get_hazards_onboarding()
        model = initialize_class(
            hazard_to_onboard, hazard_map, source_dir_base=test_output_dir
        )
        model.create_maps(source=target, target=target)
        return

    doc_store = DocStore(local_path=test_output_dir)
    onboard_hazards_to(
        target=target,
        doc_store=doc_store,
        source_dir_base=test_output_dir,
        hazards=[hazard_to_onboard],
        download_dir=str(Path(test_output_dir) / "downloads"),
    )


@pytest.mark.manual
def test_sync_named_hazard(test_output_dir, s3_credentials):
    """Used to sync a named hazard to an S3 bucket.

    Run test_onboard_named_hazard first so that the data this looks for is actually
    present in test_output_dir.
    """
    hazard_to_onboard = "JRCRiverFlood"
    dry_run = False
    maps_only = (
        False  # True to sync only the map arrays (e.g. after regenerating maps).
    )

    zarr_dir = os.path.join(test_output_dir, "hazard-indicators", "hazard.zarr")

    hazard_map = get_hazards_onboarding()
    model = initialize_class(
        hazard_to_onboard, hazard_map, source_dir_base=test_output_dir
    )

    # resource.path/resource.map.path are format strings templated with
    # scenario/year (e.g. "water_risk/wri/v2/{key}_{scenario}_{year}"); the directory
    # holding every expanded variant is the parent of that template.
    array_paths = set()
    for resource in model.inventory():
        if not maps_only:
            array_paths.add(str(PurePosixPath(resource.path).parent))
        if resource.map is not None:
            array_paths.add(str(PurePosixPath(resource.map.path).parent))

    s3_utilities.sync_local_to_prod(zarr_dir, sorted(array_paths), dry_run=dry_run)


@pytest.mark.manual
def test_backup_prod_to_local():
    """Back up the whole production hazard.zarr to local disk.

    The first run downloads everything; later runs download only files that are
    missing or changed. The (public) bucket is read anonymously, so no credentials
    are needed, and local files no longer in the bucket are kept. Run with dry_run = True first to see the number of files
    and size of the download.
    """
    dry_run = False
    zarr_dir = "/mnt/data/hazard-indicators-prod/hazard.zarr"
    s3_utilities.sync_prod_to_local(zarr_dir, dry_run=dry_run)


@pytest.mark.manual
def test_v3_heat_indicators_single_gcm(
    test_output_dir, log_to_stdout, log_to_file, log_memory
):
    """Check the v3 chronic heat indicators end to end, for a single GCM, scenario and central year.

    Runs heating/cooling degree days, days with tas/tasmax above thresholds and days with WBGT above
    thresholds (each with the multi-model mean, which for a single GCM equals that GCM), writes
    them to the local Zarr store zarr_dir and creates the map tiles. The inputs are read from NEX-GDDP-CMIP6
    files cached in cache_dir, any not in the cache being downloaded (anonymously, from the public
    bucket) when first opened; with prefetch, they are all downloaded in parallel first, which is
    faster. The log is also written to test_output/logs/, with the memory use every 30 s; while
    running, the dask dashboard (link in the log) shows the progress and memory of the tasks.
    """
    cache_dir = Path("/mnt/climatedata/climate_data/raw_downloads")
    prefetch = False
    create_maps = True
    gcm, scenario, central_year = "ACCESS-CM2", "ssp585", 2030
    # each year of WBGT holds about 6-8 GB in memory: limit the years calculated at once
    threads = 2

    run = RunConfig(gcms=[gcm], scenarios=[scenario], central_years=[central_year])
    source = NexGddpCmip6CachedSource(cache_dir)
    models: list[ThresholdBasedAverageIndicator] = [
        HeatingCoolingDegreeDays(run=run),
        DaysTasAboveIndicator(run=run),
        DaysWBGTAboveIndicator(run=run),
    ]
    if prefetch:
        years = models[0]._years(
            source, ThresholdBatchItem(gcm, scenario, central_year)
        )
        quantities = sorted({q for model in models for q in model.quantities})
        source.download_to_cache([gcm], [scenario], quantities, list(years))

    zarr_dir = "/mnt/data/hazard-indicators/hazard.zarr"
    target = OscZarr(store=zarr.DirectoryStore(zarr_dir))
    with (
        LocalCluster(
            processes=False, n_workers=1, threads_per_worker=threads
        ) as cluster,
        Client(cluster) as client,
    ):
        logging.info(f"Dask dashboard: {client.dashboard_link}")
        for model in models:
            # debug_mode: raise errors rather than logging them and continuing
            model.run_all(source, target, client=client, debug_mode=True)
            if create_maps:
                model.create_maps(target, target)

    for model in models:
        for resource in model.inventory():
            for g in [gcm, MULTI_MODEL_ID]:
                path = resource.path.format(gcm=g, scenario=scenario, year=central_year)
                da = target.read(path)
                assert da.dims[0] == "threshold"
                assert list(da["threshold"].values) == model.thresholds
                values = da.values
                assert np.isfinite(values).any(), path
                logging.info(
                    f"{path} ({da.attrs['units']}): "
                    + ", ".join(
                        f"{t}{da['threshold'].attrs['units']}: max {np.nanmax(v):.1f}"
                        for t, v in zip(model.thresholds, values, strict=True)
                    )
                )


@pytest.mark.manual
def test_v3_heat_indicators_prefetch(log_to_stdout, log_to_file):
    """Download to cache_dir all the NEX-GDDP-CMIP6 files needed for the full v3 chronic heat run.

    That is, the files of every quantity of the three v3 indicators (heating/cooling degree days,
    days with tas/tasmax above thresholds and days with WBGT above thresholds), for every GCM, scenario
    and year averaged in the default RunConfig. Files already in the cache are skipped, so this can be
    re-run (e.g. after a failure) to fetch only what is missing. Run with dry_run = True first to see
    the number of files to download.
    """
    cache_dir = Path("/mnt/climatedata/climate_data/raw_downloads")
    dry_run = False

    run = RunConfig()
    source = NexGddpCmip6CachedSource(cache_dir)
    models: list[ThresholdBasedAverageIndicator] = [
        HeatingCoolingDegreeDays(run=run),
        DaysTasAboveIndicator(run=run),
        DaysWBGTAboveIndicator(run=run),
    ]
    quantities = sorted({q for model in models for q in model.quantities})
    # a set, since the averaging periods of central years can overlap
    paths = sorted(
        {
            source._path(item.gcm, item.scenario, quantity, year)
            for item in models[0].batch_items()
            for year in models[0]._years(source, item)
            for quantity in quantities
        }
    )
    missing = [p for p in paths if not (cache_dir / PurePosixPath(p).name).exists()]
    logging.info(
        f"{len(paths)} files needed for GCMs {', '.join(run.gcms)}, quantities {', '.join(quantities)}: "
        f"{len(paths) - len(missing)} in the cache, {len(missing)} to download"
    )
    if not dry_run:
        download_paths(missing, cache_dir, source.client)


@pytest.mark.manual
def test_create_maps_from_prod_readonly(test_output_dir):
    """Generate maps locally from the array data already on the production bucket.

    Deliberately uses an anonymous (unsigned) S3FileSystem -- no access key, secret,
    or token -- so the production bucket can only ever be read here, never written
    to: any attempted write would be rejected by S3 for lack of credentials, with no
    reliance on us getting read-only permissions/IAM policy right.
    """
    hazard_to_onboard = "DavydzenkaEtAlLandSubsidence"

    anon_s3 = s3fs.S3FileSystem(anon=True)
    prod_store = s3fs.S3Map(
        root=str(
            PurePosixPath(
                "os-climate-physical-risk", "hazard-indicators", "hazard.zarr"
            )
        ),
        s3=anon_s3,
        check=False,
    )
    source = OscZarr(store=prod_store)

    local_dir = os.path.join(test_output_dir, "hazard-indicators", "hazard.zarr")
    target = OscZarr(store=zarr.DirectoryStore(local_dir))

    hazard_map = get_hazards_onboarding()
    model = initialize_class(
        hazard_to_onboard, hazard_map, source_dir_base=test_output_dir
    )

    model.create_maps(source=source, target=target)


def copy_iris_files(s3_credentials):
    bucket = os.environ["OSC_S3_BUCKET_DEV"]
    s3 = s3_utilities.get_s3_fs()
    files = [
        "/wind/IRIS/return_value_maps/IRIS_return_value_map_README.txt",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_2050-SSP1_tenthdeg.nc",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_2050-SSP2_tenthdeg.nc",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_2050-SSP5_tenthdeg.nc",
        "/wind/IRIS/return_value_maps/IRIS_vmax_maps_PRESENT_tenthdeg.nc",
    ]
    for file in files:
        parts = file.strip("/").split("/")
        filepath = os.path.join(test_output_dir, *parts)
        s3_path = str(PurePosixPath(bucket, "inputs", *parts))
        s3.put(filepath, s3_path, recursive=True)

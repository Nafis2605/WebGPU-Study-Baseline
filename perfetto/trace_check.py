"""Row counts of the tracks a Pixel GPU trace must contain (verification gate item 5).

    tools/perfetto/venv/Scripts/python perfetto/trace_check.py <trace.pftrace> [...]
Prints one JSON line per trace: GPU counter tracks and samples, GPU-frequency samples, render-stage slices, power-rail
tracks and samples, battery counters, thermal samples and Chrome track-event slices.
"""
import json
import sys

from perfetto.trace_processor import TraceProcessor

QUERIES = {
    "gpu_counter_tracks": "select count(*) from gpu_counter_track",
    "gpu_counter_samples": "select count(*) from counter c join gpu_counter_track t on c.track_id = t.id",
    "gpu_freq_samples": "select count(*) from counter c join counter_track t on c.track_id = t.id where t.name like '%gpufreq%' or t.name = 'gpufreq'",
    "render_stage_slices": "select count(*) from gpu_slice",
    "power_rail_tracks": "select count(*) from counter_track where name like 'power.%'",
    "power_rail_samples": "select count(*) from counter c join counter_track t on c.track_id = t.id where t.name like 'power.%'",
    "battery_samples": "select count(*) from counter c join counter_track t on c.track_id = t.id where t.name like 'batt.%'",
    "thermal_samples": "select count(*) from counter c join counter_track t on c.track_id = t.id where t.name like '%Temp%' or t.name like '%thermal%'",
    "chrome_slices": "select count(*) from slice s join thread_track tt on s.track_id = tt.id join thread using(utid) join process using(upid) where process.name like '%chrome%'",
    "chrome_gpu_slices": "select count(*) from slice where category like '%gpu%' or category like '%dawn%'",
}


def check(path):
    tp = TraceProcessor(trace=path)
    out = {"trace": path}
    for k, q in QUERIES.items():
        try:
            out[k] = int(tp.query(q).as_pandas_dataframe().iloc[0, 0])
        except Exception as e:  # noqa: BLE001
            out[k] = f"error: {str(e)[:80]}"
    try:
        out["gpu_counter_names"] = tp.query("select distinct name from gpu_counter_track limit 60").as_pandas_dataframe().name.tolist()
        out["power_rails"] = tp.query("select distinct name from counter_track where name like 'power.%'").as_pandas_dataframe().name.tolist()
    except Exception:  # noqa: BLE001
        pass
    tp.close()
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(json.dumps(check(p)))

// Browser scheduling helpers (from the ONNX study's timing.js).

/**
 * Give the main thread back to the browser (input, rendering) between inferences.
 * scheduler.yield() where available (Chrome 129+), otherwise a MessageChannel task (no 4 ms
 * timer clamping). Returns the mechanism used, so results record it.
 */
export async function yieldToBrowser() {
  if (globalThis.scheduler?.yield) {
    await scheduler.yield();
    return "scheduler.yield";
  }
  await new Promise(r => {
    const ch = new MessageChannel();
    ch.port1.onmessage = () => r();
    ch.port2.postMessage(null);
  });
  return "message_channel";
}

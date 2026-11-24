import { appendFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function capture(pi: ExtensionAPI) {
  const destination = process.env.INKLING_PI_REQUESTS;
  const effort = Number(process.env.INKLING_PI_EFFORT);
  const temperature = Number(process.env.INKLING_PI_TEMPERATURE ?? 0);
  if (!Number.isFinite(effort) || effort < 0 || effort > 0.99
    || !Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
    throw new Error("Invalid benchmark effort or temperature");
  }
  pi.on("before_provider_request", (event) => {
    const payload = event.payload as Record<string, unknown>;
    if (!Array.isArray(payload.messages)) throw new Error("Expected OpenAI chat messages");
    const request = { ...payload, reasoning_effort: effort, temperature };
    // Optional debug copy. Training data comes from the proxy's capture, not this file.
    if (destination) appendFileSync(destination, JSON.stringify({
      messages: payload.messages,
      tools: payload.tools ?? [],
      effort,
      temperature,
    }) + "\n", { mode: 0o600 });
    return request;
  });
}

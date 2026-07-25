import { readFile } from "node:fs/promises";
import {
  HttpCachingChain,
  HttpChainClient,
  fetchBeacon,
} from "drand-client";
import { isRetryableRemoteFailure } from "./retry-policy.mjs";

function fail(message, code = 1) {
  process.stderr.write(`${message}\n`);
  process.exit(code);
}

const inputPath = process.argv[2];
if (!inputPath) {
  fail("usage: node verify.mjs INPUT.json");
}

let input;
try {
  input = JSON.parse(await readFile(inputPath, "utf8"));
} catch (error) {
  fail(`invalid verifier input: ${error.message}`);
}

const expectedKeys = [
  "base_url",
  "chain_hash",
  "expected_pulse",
  "public_key",
];
if (
  !input ||
  typeof input !== "object" ||
  Array.isArray(input) ||
  Object.keys(input).sort().join("|") !== expectedKeys.join("|")
) {
  fail("verifier input schema mismatch");
}

const expected = input.expected_pulse;
if (!expected || !Number.isSafeInteger(expected.round) || expected.round < 1) {
  fail("expected pulse round must be a positive safe integer");
}

const options = {
  disableBeaconVerification: false,
  noCache: true,
  chainVerificationParams: {
    chainHash: input.chain_hash,
    publicKey: input.public_key,
  },
};

try {
  const chain = new HttpCachingChain(input.base_url, options);
  const client = new HttpChainClient(chain, options);
  const verified = await fetchBeacon(client, expected.round);
  const fields = ["round", "randomness", "signature", "previous_signature"];
  if (
    Object.keys(expected).sort().join("|") !== fields.slice().sort().join("|") ||
    Object.keys(verified).sort().join("|") !== fields.slice().sort().join("|") ||
    fields.some((field) => verified[field] !== expected[field])
  ) {
    fail("cryptographically verified pulse differs from frozen expected bytes");
  }
  process.stdout.write(`${JSON.stringify(verified)}\n`);
} catch (error) {
  const message = String(error?.message ?? error);
  if (isRetryableRemoteFailure(error)) {
    fail(`drand remote transport failed: ${message}`, 75);
  }
  fail(`drand cryptographic verification failed: ${message}`);
}

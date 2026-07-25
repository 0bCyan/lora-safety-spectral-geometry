export function isRetryableRemoteFailure(error) {
  const message = String(error?.message ?? error);
  return (
    message === "fetch failed" ||
    /^Error response fetching https?:\/\/\S+ - got \d+$/.test(message)
  );
}

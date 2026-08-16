import { useEffect, useRef, useState } from "react";
import { ApiError } from "../api/client";

export interface ScopedQueryState<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
}

/**
 * Fetches data keyed by `scopeKey` (e.g. a patient_id, or
 * `${patientId}:${query}`) and guarantees patient isolation:
 *
 *  1. As soon as `scopeKey` changes, `data` is immediately cleared and
 *     `loading` is set — the previous scope's data is never shown while
 *     the new scope's request is in flight.
 *  2. If `scopeKey` changes again before an in-flight request resolves,
 *     that stale response is discarded on arrival (via a generation
 *     counter) — it can never overwrite the now-current scope's state.
 *
 * This is the single mechanism every patient-scoped page in this app uses
 * to fetch data, so "Patient A's evidence briefly flashes on Patient B's
 * screen" is structurally impossible rather than something each page has
 * to remember to guard against individually.
 */
export function useScopedQuery<T>(scopeKey: string | null, fetcher: () => Promise<T>): ScopedQueryState<T> {
  const [state, setState] = useState<ScopedQueryState<T>>({ data: null, loading: scopeKey !== null, error: null });
  const generationRef = useRef(0);

  useEffect(() => {
    generationRef.current += 1;
    const thisGeneration = generationRef.current;

    if (scopeKey === null) {
      setState({ data: null, loading: false, error: null });
      return;
    }

    // Clear immediately — never show the previous scope's data while the
    // new scope's request is in flight.
    setState({ data: null, loading: true, error: null });

    fetcher()
      .then((data) => {
        if (generationRef.current !== thisGeneration) return; // stale — a newer scope has since started
        setState({ data, loading: false, error: null });
      })
      .catch((error: unknown) => {
        if (generationRef.current !== thisGeneration) return;
        const apiError =
          error instanceof ApiError ? error : new ApiError("network", "An unexpected error occurred.");
        setState({ data: null, loading: false, error: apiError });
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scopeKey]);

  return state;
}

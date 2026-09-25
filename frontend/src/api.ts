import type { DashboardSnapshot } from './types';

const apiBase = import.meta.env.VITE_API_BASE_URL ?? '';

export async function getDashboardSnapshot(signal?: AbortSignal, refresh = false): Promise<DashboardSnapshot> {
  const suffix = refresh ? '?refresh=true' : '';
  const response = await fetch(`${apiBase}/api/v1/dashboard/snapshot${suffix}`, { signal });
  if (!response.ok) throw new Error(`Dashboard API returned ${response.status}`);
  const payload: unknown = await response.json();
  if (!payload || typeof payload !== 'object' || (payload as { schema_version?: unknown }).schema_version !== '1.0') {
    throw new Error('Unsupported dashboard snapshot schema');
  }
  return payload as DashboardSnapshot;
}

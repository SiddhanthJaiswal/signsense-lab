import type { Job, Health, Rig } from './types';

export async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = body?.detail;
    throw new Error(typeof detail === 'string' ? detail : Array.isArray(detail) ? detail[0]?.msg : `Request failed (${response.status}).`);
  }
  return response.json() as Promise<T>;
}
export const getHealth = () => request<Health>('/api/health');
export const getRig = () => request<Rig>('/api/rig');
export const startTranslation = (text: string, signal: AbortSignal) => request<Job>('/api/translations', {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal,
});
export const getJob = (id: string, signal: AbortSignal) => request<Job>(`/api/translations/${id}`, { signal });

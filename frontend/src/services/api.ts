/**
 * API Client for Aldar Kose Storyboard Generation
 * 
 * Handles communication with the FastAPI backend server.
 */

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface StoryRequest {
  prompt: string;
  use_ref_guided?: boolean;
  num_frames?: number;
  seed?: number;
  gpt_temperature?: number;
}

export interface FrameResponse {
  frame_number: number;
  image: string; // Base64-encoded PNG
  prompt: string;
  clip_score: number;
}

export interface StoryResponse {
  success: boolean;
  story_prompt: string;
  num_frames: number;
  frames: FrameResponse[];
  generation_time_seconds: number;
  mode: 'simple' | 'ref-guided';
  seed: number;
  gpt_temperature: number;
  error?: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  lora_path: string;
  ref_guided_available: boolean;
}

/**
 * Check if the API server is healthy and ready
 */
export async function checkHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_URL}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    return response.json();
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('Unfortunately we could not reach your VM. Your VM IP is closed from external API requests.');
    }
    throw error;
  }
}

/**
 * Generate a story from a text prompt
 */
export async function generateStory(request: StoryRequest): Promise<StoryResponse> {
  try {
    const response = await fetch(`${API_URL}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: response.statusText }));
      throw new Error(error.error || `Generation failed: ${response.statusText}`);
    }

    return response.json();
  } catch (error) {
    if (error instanceof TypeError && error.message.includes('fetch')) {
      throw new Error('Unfortunately we cannot reach your VM. Your VM IP is closed from external API requests.');
    }
    throw error;
  }
}

/**
 * Convert base64 image string to a data URL that can be used in <img> src
 */
export function base64ToDataUrl(base64: string): string {
  return `data:image/png;base64,${base64}`;
}

/**
 * Download a base64 image as a PNG file
 */
export function downloadImage(base64: string, filename: string): void {
  const link = document.createElement('a');
  link.href = base64ToDataUrl(base64);
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Download all frames as individual PNG files
 */
export function downloadAllFrames(frames: FrameResponse[], storyName: string = 'story'): void {
  frames.forEach((frame) => {
    const filename = `${storyName}_frame_${String(frame.frame_number).padStart(3, '0')}.png`;
    downloadImage(frame.image, filename);
  });
}

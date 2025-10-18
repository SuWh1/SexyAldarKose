# Frontend API Integration

## Configuration

The frontend is configured to connect to your backend API server.

### Environment Variables

Set the API URL in `.env`:

```env
VITE_API_URL=http://154.57.34.97:8000
```

**Note:** Make sure your backend server is running at this address with CORS enabled.

## Features

### API Client (`src/services/api.ts`)

Provides TypeScript functions to interact with the backend:

- `checkHealth()` - Verify API server is running
- `generateStory(request)` - Generate a story from a prompt
- `base64ToDataUrl(base64)` - Convert base64 to image data URL
- `downloadImage(base64, filename)` - Download a single frame
- `downloadAllFrames(frames, storyName)` - Download all frames as PNG files

### Story Generator Component

Navigate to `/generate` to access the story generation interface.

**Features:**
- Text prompt input
- Reference-guided mode toggle
- Number of frames control (6-10)
- Seed for reproducibility
- GPT temperature control (0.0-1.0)
- Real-time generation progress
- Frame preview with CLIP scores
- Download individual frames or all at once

### Request Parameters

```typescript
{
  prompt: string;              // Story description (required)
  use_ref_guided?: boolean;    // Better face consistency (default: false)
  num_frames?: number;         // 6-10 frames, or auto (default: auto)
  seed?: number;               // Random seed (default: 42)
  gpt_temperature?: number;    // 0.0-1.0, creativity level (default: 0.7)
}
```

### Response Format

```typescript
{
  success: boolean;
  story_prompt: string;
  num_frames: number;
  frames: [
    {
      frame_number: number;
      image: string;           // Base64-encoded PNG
      prompt: string;          // Scene description
      clip_score: number;      // Quality score
    }
  ];
  generation_time_seconds: number;
  mode: "simple" | "ref-guided";
  seed: number;
  gpt_temperature: number;
}
```

## Usage Flow

1. **Home Page** - User enters a story prompt in the hero section
2. **Navigate** - Clicking "Generate" redirects to `/generate` with the prompt
3. **Auto-Generate** - Story generation starts automatically
4. **View Results** - Frames displayed in vertical comic layout
5. **Download** - User can download individual frames or all at once

## Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Server Requirements

Your backend server must:
1. Be running at the configured `VITE_API_URL`
2. Have CORS enabled (already configured in `api/server.py`)
3. Have the following endpoints:
   - `GET /health` - Health check
   - `POST /generate` - Story generation

## Example API Call

```typescript
import { generateStory } from './services/api';

const result = await generateStory({
  prompt: "Aldar Kose tricks a wealthy merchant",
  use_ref_guided: false,
  seed: 42,
  gpt_temperature: 0.7,
  num_frames: 8
});

console.log(`Generated ${result.num_frames} frames in ${result.generation_time_seconds}s`);
```

## CORS Configuration

The backend server is already configured with CORS to allow requests from any origin:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Troubleshooting

### API Connection Failed

1. Verify backend server is running:
   ```bash
   curl http://154.57.34.97:8000/health
   ```

2. Check `.env` file has correct URL with protocol and port:
   ```env
   VITE_API_URL=http://154.57.34.97:8000
   ```

3. Restart Vite dev server after changing `.env`:
   ```bash
   npm run dev
   ```

### Generation Takes Too Long

- Simple mode: ~4 minutes for 8 frames
- Ref-guided mode: ~5 minutes for 8 frames
- This is normal - the UI shows a loading spinner

### Images Not Displaying

Check that the base64 data is properly converted:
```typescript
import { base64ToDataUrl } from './services/api';
const dataUrl = base64ToDataUrl(frame.image);
```

## Development Tips

1. **Type Safety**: All API types are defined in `src/services/api.ts`
2. **Error Handling**: The API client throws errors that can be caught
3. **Loading States**: Use the `loading` state in components
4. **Auto-Generation**: Prompt from navigation state triggers auto-generation

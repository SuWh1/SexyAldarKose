import { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { generateStory, base64ToDataUrl, downloadAllFrames, type StoryResponse, type FrameResponse } from '../services/api';

export function StoryGenerator() {
  const location = useLocation();
  const initialPrompt = (location.state as { prompt?: string })?.prompt || '';
  
  const [prompt, setPrompt] = useState(initialPrompt);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [story, setStory] = useState<StoryResponse | null>(null);
  const [useRefGuided, setUseRefGuided] = useState(false);
  const [seed, setSeed] = useState<number>(42);
  const [temperature, setTemperature] = useState<number>(0.7);
  const [numFrames, setNumFrames] = useState<number | undefined>(undefined);

  // Auto-generate if prompt comes from navigation
  useEffect(() => {
    if (initialPrompt && !loading && !story && !error) {
      handleGenerate();
    }
  }, []);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a story prompt');
      return;
    }

    setLoading(true);
    setError(null);
    setStory(null);

    try {
      const result = await generateStory({
        prompt: prompt.trim(),
        use_ref_guided: useRefGuided,
        seed,
        gpt_temperature: temperature,
        num_frames: numFrames,
      });

      setStory(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate story');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (story) {
      const storyName = prompt.slice(0, 30).replace(/[^a-z0-9]/gi, '_').toLowerCase();
      downloadAllFrames(story.frames, storyName);
    }
  };

  return (
    <div className="min-h-screen" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      <section className="py-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {/* Title */}
          <div className="text-center mb-12">
            <h1 
              className="text-5xl md:text-7xl font-bold mb-6 leading-tight transition-colors duration-200" 
              style={{ color: 'rgba(209, 254, 23, 1)' }}
            >
              Generate Your Story
            </h1>
            <p className="text-lg md:text-xl" style={{ color: 'rgba(255, 255, 255, 0.7)' }}>
              Create custom Aldar Köse adventures with AI
            </p>
          </div>

          {/* Generation Form */}
          <div className="max-w-3xl mx-auto mb-12">
            <div className="rounded-2xl p-8" style={{ backgroundColor: 'rgba(27, 29, 17, 1)' }}>
              {/* Prompt Input */}
              <div className="mb-6">
                <label className="block text-lg font-semibold mb-2" style={{ color: 'rgba(209, 254, 23, 1)' }}>
                  Story Prompt
                </label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="e.g., Aldar Kose tricks a wealthy merchant and steals his horse..."
                  className="w-full px-4 py-3 rounded-lg resize-none focus:outline-none focus:ring-2"
                  style={{ 
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                    color: 'rgba(255, 255, 255, 1)',
                    border: '2px solid rgba(209, 254, 23, 0.3)',
                    minHeight: '120px'
                  }}
                  disabled={loading}
                />
              </div>

              {/* Settings Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                {/* Mode Toggle */}
                <div>
                  <label className="flex items-center gap-3 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={useRefGuided}
                      onChange={(e) => setUseRefGuided(e.target.checked)}
                      disabled={loading}
                      className="w-5 h-5 rounded"
                      style={{ accentColor: 'rgba(209, 254, 23, 1)' }}
                    />
                    <span style={{ color: 'rgba(255, 255, 255, 0.9)' }}>
                      Use Reference-Guided Mode
                      <span className="block text-sm" style={{ color: 'rgba(255, 255, 255, 0.5)' }}>
                        Better face consistency
                      </span>
                    </span>
                  </label>
                </div>

                {/* Num Frames */}
                <div>
                  <label className="block text-sm font-semibold mb-2" style={{ color: 'rgba(209, 254, 23, 1)' }}>
                    Frames (optional)
                  </label>
                  <input
                    type="number"
                    value={numFrames || ''}
                    onChange={(e) => setNumFrames(e.target.value ? parseInt(e.target.value) : undefined)}
                    placeholder="Auto (GPT decides)"
                    min="6"
                    max="10"
                    disabled={loading}
                    className="w-full px-4 py-2 rounded-lg focus:outline-none focus:ring-2"
                    style={{ 
                      backgroundColor: 'rgba(0, 0, 0, 0.5)',
                      color: 'rgba(255, 255, 255, 1)',
                      border: '2px solid rgba(209, 254, 23, 0.3)'
                    }}
                  />
                </div>

                {/* Seed */}
                <div>
                  <label className="block text-sm font-semibold mb-2" style={{ color: 'rgba(209, 254, 23, 1)' }}>
                    Seed (for reproducibility)
                  </label>
                  <input
                    type="number"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
                    disabled={loading}
                    className="w-full px-4 py-2 rounded-lg focus:outline-none focus:ring-2"
                    style={{ 
                      backgroundColor: 'rgba(0, 0, 0, 0.5)',
                      color: 'rgba(255, 255, 255, 1)',
                      border: '2px solid rgba(209, 254, 23, 0.3)'
                    }}
                  />
                </div>

                {/* Temperature */}
                <div>
                  <label className="block text-sm font-semibold mb-2" style={{ color: 'rgba(209, 254, 23, 1)' }}>
                    GPT Temperature: {temperature.toFixed(1)}
                  </label>
                  <input
                    type="range"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    min="0"
                    max="1"
                    step="0.1"
                    disabled={loading}
                    className="w-full"
                    style={{ accentColor: 'rgba(209, 254, 23, 1)' }}
                  />
                  <div className="flex justify-between text-xs mt-1" style={{ color: 'rgba(255, 255, 255, 0.5)' }}>
                    <span>Deterministic</span>
                    <span>Creative</span>
                  </div>
                </div>
              </div>

              {/* Generate Button */}
              <button
                onClick={handleGenerate}
                disabled={loading || !prompt.trim()}
                className="w-full px-8 py-4 rounded-xl text-lg font-semibold transition-all duration-200 hover:brightness-90 disabled:opacity-50 disabled:cursor-not-allowed"
                style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
              >
                {loading ? 'Generating...' : 'Generate Story'}
              </button>

              {/* Error Display */}
              {error && (
                <div className="mt-4 p-4 rounded-lg" style={{ backgroundColor: 'rgba(255, 0, 0, 0.1)', border: '1px solid rgba(255, 0, 0, 0.3)' }}>
                  <p style={{ color: 'rgba(255, 100, 100, 1)' }}>❌ {error}</p>
                </div>
              )}

              {/* Loading Progress */}
              {loading && (
                <div className="mt-6 text-center">
                  <div className="inline-block animate-spin rounded-full h-12 w-12 border-4 border-t-transparent" style={{ borderColor: 'rgba(209, 254, 23, 1)', borderTopColor: 'transparent' }}></div>
                  <p className="mt-4" style={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                    Generating your story... This may take 4-5 minutes
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Results Display */}
          {story && (
            <div className="max-w-4xl mx-auto">
              <div className="rounded-2xl p-8 mb-8" style={{ backgroundColor: 'rgba(27, 29, 17, 1)' }}>
                {/* Story Info */}
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h2 className="text-2xl font-bold mb-2" style={{ color: 'rgba(209, 254, 23, 1)' }}>
                      {story.story_prompt}
                    </h2>
                    <div className="flex gap-4 text-sm" style={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                      <span>🎬 {story.num_frames} frames</span>
                      <span>⏱️ {Math.round(story.generation_time_seconds)}s</span>
                      <span>🎨 {story.mode}</span>
                      <span>🎲 Seed: {story.seed}</span>
                      <span>🌡️ Temp: {story.gpt_temperature}</span>
                    </div>
                  </div>
                  <button
                    onClick={handleDownload}
                    className="px-6 py-3 rounded-xl font-semibold transition-all duration-200 hover:brightness-90"
                    style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
                  >
                    Download All
                  </button>
                </div>

                {/* Frames Grid */}
                <div className="flex flex-col gap-6">
                  {story.frames.map((frame: FrameResponse) => (
                    <div key={frame.frame_number} className="rounded-lg overflow-hidden" style={{ border: '2px solid rgba(209, 254, 23, 0.3)' }}>
                      <img
                        src={base64ToDataUrl(frame.image)}
                        alt={`Frame ${frame.frame_number}: ${frame.prompt}`}
                        className="w-full h-auto"
                      />
                      <div className="p-4" style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}>
                        <div className="flex justify-between items-start">
                          <p style={{ color: 'rgba(255, 255, 255, 0.9)' }}>
                            <span className="font-bold" style={{ color: 'rgba(209, 254, 23, 1)' }}>
                              Frame {frame.frame_number}:
                            </span>{' '}
                            {frame.prompt}
                          </p>
                          <span className="text-sm whitespace-nowrap ml-4" style={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                            CLIP: {frame.clip_score.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Back Button */}
              <div className="flex justify-center">
                <a 
                  href="/"
                  className="px-8 py-4 rounded-2xl text-lg font-semibold transition-all duration-200 hover:brightness-90 shadow-lg whitespace-nowrap flex items-center gap-2"
                  style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
                >
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 5L7 10L12 15" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  Back to Home
                </a>
              </div>
            </div>
          )}
        </div>
      </section>
    </div>
  );
}

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { LoaderCircle } from 'lucide-react';

export function HackathonHero() {
  const [prompt, setPrompt] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  // Auto-clear error after 5 seconds
  useEffect(() => {
    if (error) {
      const timer = setTimeout(() => {
        setError('');
      }, 5000);

      return () => clearTimeout(timer);
    }
  }, [error]);

  const handleGenerate = async () => {
    // Clear any previous error
    setError('');
    
    // Validate input
    if (!prompt.trim()) {
      setError('Please describe a scene with Aldar Köse to generate the storyboard');
      return;
    }
    
    if (!isLoading) {
      setIsLoading(true);
      
      // Simulate API call with 3 second delay
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Navigate to the comics page
      navigate('/aldarkose');
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleGenerate();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setPrompt(e.target.value);
    // Clear error when user starts typing
    if (error) {
      setError('');
    }
  };

  return (
    <section className="py-12 animate-in fade-in duration-1000" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center">
          {/* Badge */}
          <div className="mb-8 animate-in slide-in-from-top duration-700 delay-200">
            <a 
              href="https://higgsfield.ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-block px-4 py-1.5 rounded-full text-xs sm:text-sm font-semibold transition-all duration-200 hover:brightness-90 cursor-pointer" 
              style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
            >
              Higgsfield AI Hackathon 2025
            </a>
          </div>

          {/* Main Heading */}
          <h1 className="text-5xl md:text-7xl font-bold mb-8 leading-tight cursor-pointer transition-colors duration-200" 
            style={{ color: 'rgba(209, 254, 23, 1)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(180, 220, 20, 1)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
          >
            <span className="inline-block">
              Aldar Köse
            </span><br />
            <span className="inline-block">
              Storyboard Generator
            </span>
          </h1>

          {/* Subheading */}
          <h2 className="text-2xl md:text-3xl font-semibold mb-12 cursor-pointer transition-colors duration-200" 
            style={{ color: 'rgba(247, 247, 248, 1)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
          >
            From script to storyboard in seconds
          </h2>

          {/* Prompt Input */}
          <div className="flex flex-col items-center my-12 max-w-4xl mx-auto animate-in slide-in-from-bottom duration-800 delay-700">
            <div className="flex flex-col sm:flex-row gap-4 justify-center w-full">
              <input 
                type="text"
                placeholder="Describe the scene with Aldar Kose"
                value={prompt}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                disabled={isLoading}
                className="flex-1 px-6 py-4 rounded-2xl text-lg transition-all duration-200 placeholder:text-[rgba(160,160,160,1)]"
                style={{ 
                  backgroundColor: 'rgba(17, 17, 17, 1)', 
                  color: 'rgba(247, 247, 248, 1)',
                  border: error ? '1px solid rgba(239, 68, 68, 1)' : '1px solid rgba(27, 29, 17, 1)',
                  outline: 'none',
                  opacity: isLoading ? 0.7 : 1
                }}
              />
              <button 
                onClick={handleGenerate}
                disabled={isLoading}
                className="px-8 py-4 rounded-2xl text-lg font-semibold transition-all duration-200 hover:brightness-90 shadow-lg whitespace-nowrap flex items-center gap-2"
                style={{ 
                  backgroundColor: 'rgba(209, 254, 23, 1)', 
                  color: 'rgba(9, 13, 14, 1)',
                  opacity: isLoading ? 0.8 : 1,
                  cursor: isLoading ? 'not-allowed' : 'pointer'
                }}
              >
                Generate
                {isLoading ? (
                  <LoaderCircle className="animate-spin" size={20} />
                ) : (
                  <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M11.8525 4.21651L11.7221 3.2387C11.6906 3.00226 11.4889 2.82568 11.2504 2.82568C11.0118 2.82568 10.8102 3.00226 10.7786 3.23869L10.6483 4.21651C10.2658 7.0847 8.00939 9.34115 5.14119 9.72358L4.16338 9.85396C3.92694 9.88549 3.75037 10.0872 3.75037 10.3257C3.75037 10.5642 3.92694 10.7659 4.16338 10.7974L5.14119 10.9278C8.00938 11.3102 10.2658 13.5667 10.6483 16.4349L10.7786 17.4127C10.8102 17.6491 11.0118 17.8257 11.2504 17.8257C11.4889 17.8257 11.6906 17.6491 11.7221 17.4127L11.8525 16.4349C12.2349 13.5667 14.4913 11.3102 17.3595 10.9278L18.3374 10.7974C18.5738 10.7659 18.7504 10.5642 18.7504 10.3257C18.7504 10.0872 18.5738 9.88549 18.3374 9.85396L17.3595 9.72358C14.4913 9.34115 12.2349 7.0847 11.8525 4.21651Z" fill="currentColor"></path>
                  </svg>
                )}
              </button>
            </div>
            
            {/* Error Message */}
            {error && (
              <div className="mt-3 text-left w-full animate-in fade-in slide-in-from-top duration-300">
                <p className="text-sm font-medium" style={{ color: 'rgba(239, 68, 68, 1)' }}>
                  {error}
                </p>
              </div>
            )}
          </div>

          {/* Description */}
          <p className="text-base md:text-lg max-w-4xl mx-auto leading-relaxed cursor-pointer transition-colors duration-200" 
            style={{ color: 'rgba(153, 153, 153, 1)' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(167, 203, 18, 1)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(153, 153, 153, 1)'}
          >
            An intelligent system that automatically generates 6-10 frame storyboards from 
            short scripts, bringing Aldar Köse's adventures to life with AI-powered visual storytelling.
          </p>
        </div>
      </div>
    </section>
  );
}
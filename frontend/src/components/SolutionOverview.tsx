export function SolutionOverview() {
  return (
    <section id="solution" className="py-20" style={{ backgroundColor: 'rgba(0, 0, 0, 1)' }}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          {/* Left Content */}
          <div className="animate-in slide-in-from-left duration-800">
            <h2 id="approach" className="text-4xl md:text-5xl font-bold mb-8 leading-tight pt-40 cursor-pointer" 
              style={{ marginTop: '-13rem', color: 'rgba(209, 254, 23, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(180, 220, 20, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
            >
              Our Approach
            </h2>
            
            <div className="space-y-6 text-lg">
              <p className="cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                We built a multi-stage AI pipeline that transforms short scripts 
                into vivid storyboards with consistent character design and cultural authenticity.
              </p>
              
              <p className="cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                Our system combines LLMs, diffusion models, and ControlNet to 
                ensure Aldar Köse maintains his identity across every frame.
              </p>
              
              <p className="cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                From prompt to frame - powered by cutting-edge generative AI that respects tradition while embracing innovation.
              </p>
            </div>
          </div>

          {/* Right Content - Key Features */}
          <div className="space-y-6">
            <div className="p-6 rounded-xl transition-all duration-200 cursor-pointer group animate-in slide-in-from-right duration-800 delay-200">
              <h3 className="text-xl font-bold mb-3 transition-all duration-200"
                style={{ color: 'rgba(209, 254, 23, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(180, 220, 20, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              >
                Character Consistency
              </h3>
              <p className="cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                Aldar Köse retains his recognizable features across all frames using LoRA & IP-Adapter
              </p>
            </div>

            <div className="p-6 rounded-xl transition-all duration-200 cursor-pointer group animate-in slide-in-from-right duration-800 delay-400">
              <h3 className="text-xl font-bold mb-3 transition-all duration-200"
                style={{ color: 'rgba(209, 254, 23, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(180, 220, 20, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              >
                LLM-Driven Flow
              </h3>
              <p className="cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                Transforms 2-4 sentence scripts into detailed, frame-by-frame visual scenes
              </p>
            </div>

            <div className="p-6 rounded-xl transition-all duration-200 cursor-pointer group animate-in slide-in-from-right duration-800 delay-600">
              <h3 className="text-xl font-bold mb-3 transition-all duration-200"
                style={{ color: 'rgba(209, 254, 23, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(180, 220, 20, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              >
                Cultural Respect
              </h3>
              <p className="cursor-pointer transition-colors duration-200"
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                Honoring Kazakh folklore with authentic tone, setting, and costume design
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
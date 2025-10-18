import { useState, useEffect } from 'react';

export function HackathonHeader() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  const handleNavigation = (section: string) => {
    setIsMobileMenuOpen(false); // Close mobile menu after navigation
    const element = document.getElementById(section);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth' });
    }
  };

  useEffect(() => {
    let ticking = false;
    
    const handleScroll = () => {
      const isScrolled = window.scrollY > 100;
      if (isScrolled !== scrolled) {
        setScrolled(isScrolled);
      }
    };

    const throttledScroll = () => {
      if (!ticking) {
        requestAnimationFrame(() => {
          handleScroll();
          ticking = false;
        });
        ticking = true;
      }
    };

    window.addEventListener('scroll', throttledScroll, { passive: true });
    return () => {
      window.removeEventListener('scroll', throttledScroll);
    };
  }, [scrolled]);

  return (
    <div className="sticky top-0 z-[1100] w-full py-2">
      <div className="w-full grid place-items-center">
        <div className={`transition-all duration-300 ease-out ${
          scrolled ? 'max-w-4xl' : 'max-w-7xl'
        } w-full px-4 sm:px-6 lg:px-8`}>
          <header 
            className={`transition-all duration-300 ease-out ${
              scrolled
                ? 'backdrop-blur-md rounded-full py-3 px-8 shadow-lg border' 
                : 'backdrop-blur-none rounded-none py-2 px-8 border-b'
            }`}
            style={{ 
              backgroundColor: scrolled ? 'rgba(0, 0, 0, 0.7)' : 'rgba(0, 0, 0, 1)', 
              borderColor: 'rgba(153, 153, 153, 0.2)' 
            }}
          >
            <div className={`flex items-center transition-all duration-200 ${
              scrolled ? 'justify-between' : 'justify-between'
            }`}>
          {/* Logo */}
          <div className="flex-shrink-0">
            <a href="/" className="flex items-center gap-3 group cursor-pointer">
              <img 
                src="/icon.png" 
                alt="SexyAldarKose" 
                className={`object-contain transition-all duration-300 ${
                  scrolled ? 'h-8 w-8' : 'h-10 w-10'
                }`}
              />
              <h1 className={`font-bold transition-all duration-300 ${
                scrolled ? 'text-xl' : 'text-2xl'
              }`} 
                style={{ color: 'rgba(247, 247, 248, 1)' }}
                onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
                onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(247, 247, 248, 1)'}
              >
                SexyAldarKose
              </h1>
            </a>
          </div>

          {/* Desktop Navigation */}
          <nav className={`hidden md:flex items-center transition-all duration-300 ${
            scrolled ? 'space-x-1' : 'space-x-2'
          }`}>
            <button 
              onClick={() => handleNavigation('approach')}
              className={`font-medium transition-all duration-300 rounded-lg cursor-pointer ${
                scrolled ? 'px-4 py-2 text-sm' : 'px-6 py-3 text-base'
              }`}
              style={{ color: 'rgba(153, 153, 153, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(153, 153, 153, 1)'}
            >
              Approach
            </button>
            <button 
              onClick={() => handleNavigation('team')}
              className={`font-medium transition-all duration-300 rounded-lg cursor-pointer ${
                scrolled ? 'px-4 py-2 text-sm' : 'px-6 py-3 text-base'
              }`}
              style={{ color: 'rgba(153, 153, 153, 1)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'rgba(209, 254, 23, 1)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'rgba(153, 153, 153, 1)'}
            >
              Team
            </button>
          </nav>

          {/* Desktop CTA */}
          <div className="hidden md:flex">
            <a 
              href="https://higgsfield.ai/" 
              target="_blank" 
              rel="noopener noreferrer"
              className={`rounded-lg font-medium transition-all duration-300 hover:brightness-90 cursor-pointer ${
                scrolled ? 'px-3 py-1.5 text-xs' : 'px-4 py-2 text-sm'
              }`}
              style={{ backgroundColor: 'rgba(209, 254, 23, 1)', color: 'rgba(9, 13, 14, 1)' }}
            >
              Higgsfield AI
            </a>
          </div>

          {/* Mobile hamburger button */}
          <div className="md:hidden">
            <button
              onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
              className="p-2 rounded-md transition-all duration-300"
              style={{ color: 'rgba(153, 153, 153, 1)' }}
              aria-label="Toggle mobile menu"
            >
              <svg
                className={`h-6 w-6 transition-transform duration-300 ${isMobileMenuOpen ? 'rotate-90' : ''}`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                {isMobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>
            </div>
          </header>
        </div>
      </div>
    </div>
  );
}

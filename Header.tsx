import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

interface HeaderProps {
  className?: string;
  static?: boolean;
}

const Header: React.FC<HeaderProps> = ({ className = '', static: isStatic = false }) => {
  const [visible, setVisible] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();

  const handleScroll = () => {
    if (isStatic) return; // Don't update state if header is static
    const scrolled = window.scrollY > 100;
    if (scrolled !== visible) {
      setVisible(scrolled);
    }
  };

  useEffect(() => {
    if (isStatic) return; // Don't add scroll listener if static
    
    let ticking = false;
    
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
  }, [visible, isStatic]);

  return (
    <div className={`sticky inset-x-0 py-4 top-0 z-50 w-full ${className}`}>
      <div className="w-full grid place-items-center">
        <div className={`transition-all duration-300 ease-out ${
          (!isStatic && visible) ? 'max-w-4xl' : 'max-w-[1632px]'
        } w-full`}>
          <header 
            className={`transition-all duration-300 ease-out ${
              (!isStatic && visible)
                ? 'backdrop-blur-md bg-white/30 rounded-full py-3 px-8 shadow-lg mx-auto' 
                : 'backdrop-blur-none bg-transparent border-0 rounded-none py-4 px-8'
            }`}
          >
            <div className={`flex items-center transition-all duration-200 ${
              (!isStatic && visible) ? 'justify-between px-4' : 'justify-between'
            }`}>
              {/* Logo */}
              <div className="flex items-center">
                <a 
                  href="#" 
                  onClick={(e) => {
                    e.preventDefault();
                    if (location.pathname === '/') {
                      window.scrollTo({ top: 0, behavior: 'smooth' });
                    } else {
                      navigate('/');
                    }
                  }}
                  className={`font-bold text-gray-900 transition-all duration-200 hover:text-emerald-600 cursor-pointer ${
                    (!isStatic && visible) ? 'text-lg' : 'text-xl'
                  }`}
                >
                  <span>Team </span>
                  <span className="text-emerald-600">Richards</span>
                </a>
              </div>

              {/* Navigation - always visible but transforms */}
              <nav className={`flex items-center transition-spacing duration-200 ${
                (!isStatic && visible) ? 'space-x-4' : 'space-x-8'
              }`}>
                <a href="#about" className={`text-gray-600 hover:text-emerald-600 transition-colors font-medium ${
                  visible ? 'text-sm' : 'text-sm'
                }`}>
                  About
                </a>
                <a href="#features" className={`text-gray-600 hover:text-emerald-600 transition-colors font-medium ${
                  visible ? 'text-sm' : 'text-sm'
                }`}>
                  Features
                </a>
                <a href="#team" className={`text-gray-600 hover:text-emerald-600 transition-colors font-medium ${
                  visible ? 'text-sm' : 'text-sm'
                }`}>
                  Team
                </a>
                <button 
                  onClick={() => navigate('/solution')}
                  className={`bg-emerald-600 text-white rounded-lg font-medium hover:bg-emerald-700 transition-colors duration-200 ${
                    visible ? 'px-4 py-2 text-sm' : 'px-4 py-2 text-sm'
                  }`}
                >
                  Solution
                </button>
              </nav>
            </div>
          </header>
        </div>
      </div>
    </div>
  );
};

export default Header;
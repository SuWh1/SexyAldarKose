import type { ReactNode } from 'react';

interface HighlightProps {
  children: ReactNode;
  className?: string;
}

export function Highlight({ children, className = '' }: HighlightProps) {
  return (
    <span className={`relative inline-block ${className}`} style={{ zIndex: 1 }}>
      <span className="relative z-10 font-bold" style={{ color: '#141414' }}>
        {children}
      </span>
      
      {/* First marker stroke - covers bottom-left to top-right */}
      <span 
        className="absolute z-0 block"
        style={{
          backgroundColor: '#C1F21D',
          top: '0.45em',
          left: '-0.2em',
          right: '0.3em',
          bottom: '-0.25em',
          transform: 'skew(-2deg, 0.5deg) rotate(-0.3deg)',
          clipPath: 'ellipse(100% 45% at 50% 25%)'
        }}
      ></span>
      
      {/* Second marker stroke - covers top-left to bottom-right */}
      <span 
        className="absolute z-0 block"
        style={{
          backgroundColor: '#C1F21D',
          top: '0.2em',
          left: '0.25em',
          right: '-0.1em',
          bottom: '0.3em',
          transform: 'skew(1deg, -0.5deg) rotate(0.3deg)',
          clipPath: 'polygon(0% 0%, 25% 10%, 75% 10%, 100% 0%, 100% 100%, 0% 100%)'
        }}
      ></span>
    </span>
  );
}

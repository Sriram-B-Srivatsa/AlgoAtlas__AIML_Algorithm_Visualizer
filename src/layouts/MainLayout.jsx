import React, { useEffect, useRef } from 'react';
import Navbar from '../components/Navbar';
import Footer from '../components/Footer';

function MainLayout({ children }) {
  const layoutRef = useRef(null);

  useEffect(() => {
    if (!layoutRef.current) return;

    // A Global Observer that watches for new elements being added to the screen
    const observer = new MutationObserver((mutations) => {
      let shouldScroll = false;

      mutations.forEach((mutation) => {
        if (mutation.addedNodes) {
          mutation.addedNodes.forEach((node) => {
            // If a new DIV is added that contains our Results headers
            if (node.nodeType === 1 && node.tagName === 'DIV') {
              const text = node.textContent || '';
              if (text.includes('Results') || text.includes('Analytical Dashboard')) {
                 // Check if it's a large container (to ignore small tooltips)
                 if (node.clientHeight > 200 || node.scrollHeight > 200) {
                    shouldScroll = true;
                 }
              }
            }
          });
        }
      });

      if (shouldScroll) {
        // Scroll down just enough to reveal the top of the 2x2 grid, not the bottom!
        setTimeout(() => {
          window.scrollBy({ top: window.innerHeight * 0.65, behavior: 'smooth' });
        }, 150);
      }
    });

    observer.observe(layoutRef.current, { childList: true, subtree: true });
    return () => observer.disconnect();
  }, []);

  return (
    <div className="main-layout" ref={layoutRef}>
      <Navbar />
      <main className="main-content">
        {children}
      </main>
      <Footer />
    </div>
  );
}

export default MainLayout;

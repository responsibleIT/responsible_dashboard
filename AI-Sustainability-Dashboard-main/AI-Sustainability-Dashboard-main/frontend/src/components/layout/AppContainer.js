import React from 'react';
import { Container } from 'react-bootstrap';

/**
 * Main container component that wraps all pages
 * Sets height to 100vh to prevent scrolling
 * Uses the specified background color
 */
const AppContainer = ({ children }) => {
  return (
    <Container 
      className="p-0" 
      style={{ 
        backgroundColor: '#FDFDF7'
      }}
    >
      {children}
    </Container>
  );
};

export default AppContainer;
import React from 'react';

/**
 * Animated loading spinner component
 * Shows a circular loader with text inside the circle
 * Uses the specified background color
 */
const LoadingSpinner = ({ message }) => {
  const spinnerStyle = {
    width: '250px',
    height: '250px',
    position: 'relative',
    margin: '0 auto'
  };

  const outerCircleStyle = {
    width: '100%',
    height: '100%',
    borderRadius: '50%',
    border: '10px solid #f3f3f3',
    position: 'absolute'
  };

  const innerCircleStyle = {
    width: '100%',
    height: '100%',
    borderRadius: '50%',
    border: '10px solid transparent',
    borderTopColor: '#000',
    position: 'absolute',
    animation: 'spin 1s linear infinite'
  };

  const textContainerStyle = {
    position: 'absolute',
    top: '0',
    left: '0',
    width: '100%',
    height: '100%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    textAlign: 'center',
    padding: '0 30px',
    backgroundColor: 'transparent'
  };

  return (
    <div className="text-center mb-4">
      <div style={spinnerStyle}>
        <div style={outerCircleStyle}></div>
        <div style={innerCircleStyle}></div>
        
        <div style={textContainerStyle}>
          <p className="mb-0 fs-5">{message}</p>
        </div>
        
        <style>
          {`
            @keyframes spin {
              0% { transform: rotate(0deg); }
              100% { transform: rotate(360deg); }
            }
          `}
        </style>
      </div>
    </div>
  );
};

export default LoadingSpinner;
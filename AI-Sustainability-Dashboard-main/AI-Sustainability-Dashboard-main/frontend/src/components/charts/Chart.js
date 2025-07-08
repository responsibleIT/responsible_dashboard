import React from 'react';

/**
 * Chart component for visualizing metrics vs threshold
 * Creates a full-size chart that fills its container
 */
const Chart = ({ type, threshold }) => {
  // Define color schemes for different chart types
  const colorSchemes = {
    power: { line: '#8884d8', point: '#8884d8', axis: '#666' },
    accuracy: { line: '#82ca9d', point: '#82ca9d', axis: '#666' },
    carbon: { line: '#ffc658', point: '#ffc658', axis: '#666' },
    computing: { line: '#ff8042', point: '#ff8042', axis: '#666' }
  };
  
  const colors = colorSchemes[type] || colorSchemes.power;
  
  // Create data points (this would be real data in a production app)
  const generateDataPoints = () => {
    // Initial values for each chart type
    const initialValues = {
      power: 0.41,
      accuracy: 95,
      carbon: 75,
      computing: 32
    };
    
    // Step decreases for each chart type
    const decreases = {
      power: 0.01,
      accuracy: 1.8,
      carbon: 1.7,
      computing: 0.8
    };

    const initialValue = initialValues[type] || 100;
    const decrease = decreases[type] || 1;
    
    // Generate 20 points with slight variations
    return Array.from({ length: 20 }, (_, i) => {
      // Add some randomness for a more natural curve
      const randomFactor = 1 + (Math.random() * 0.1 - 0.05);
      const value = initialValue - (decrease * i * randomFactor);
      return {
        x: i + 1, // Threshold from 1 to 20
        y: Math.max(0, value) // Ensure no negative values
      };
    });
  };
  
  const dataPoints = generateDataPoints();
  
  // Create the SVG path from data points
  const createPathD = (points) => {
    // Scale values to fit in our 300×150 viewBox
    const scaledPoints = points.map(point => ({
      x: (point.x - 1) * (300 / 19), // Scale x from 1-20 to 0-300
      y: 150 - (point.y / (dataPoints[0].y * 1.1)) * 150 // Scale y to 0-150, inverted for SVG
    }));
    
    return scaledPoints.map((point, i) => 
      (i === 0 ? 'M' : 'L') + point.x + ',' + point.y
    ).join(' ');
  };
  
  return (
    <div className="w-100 h-100">
      <svg 
        viewBox="0 0 300 180" 
        style={{ 
          width: '100%', 
          height: '100%', 
          overflow: 'visible',
          display: 'block'
        }}
        preserveAspectRatio="none"
      >
        {/* Background grid */}
        {Array.from({ length: 6 }, (_, i) => (
          <line 
            key={`horizontal-${i}`}
            x1="0" 
            y1={i * 30} 
            x2="300" 
            y2={i * 30} 
            stroke="#eee" 
            strokeWidth="1" 
            strokeDasharray="3,3"
          />
        ))}
        
        {Array.from({ length: 21 }, (_, i) => (
          <line 
            key={`vertical-${i}`}
            x1={i * (300/20)} 
            y1="0" 
            x2={i * (300/20)} 
            y2="150" 
            stroke="#eee" 
            strokeWidth="1" 
            strokeDasharray="3,3"
          />
        ))}
        
        {/* X axis (bottom) */}
        <line x1="0" y1="150" x2="300" y2="150" stroke={colors.axis} strokeWidth="1" />
        
        {/* Y axis (left) */}
        <line x1="0" y1="0" x2="0" y2="150" stroke={colors.axis} strokeWidth="1" />
        
        {/* Threshold vertical line */}
        <line 
          x1={(threshold - 1) * (300 / 19)} 
          y1="0" 
          x2={(threshold - 1) * (300 / 19)} 
          y2="150" 
          stroke="red" 
          strokeWidth="1.5" 
        />
        
        {/* Chart line */}
        <path
          d={createPathD(dataPoints)}
          fill="none"
          stroke={colors.line}
          strokeWidth="1.5"
        />
        
        {/* Data points */}
        {dataPoints.map((point, i) => {
          const x = (point.x - 1) * (300 / 19);
          const y = 150 - (point.y / (dataPoints[0].y * 1.1)) * 150;
          
          return (
            <circle
              key={i}
              cx={x}
              cy={y}
              r="3"
              fill="#fff"
              stroke={colors.point}
              strokeWidth="1.5"
            />
          );
        })}
        
        {/* X-axis labels */}
        {[1, 5, 10, 15, 20].map(num => (
          <text 
            key={`x-label-${num}`}
            x={(num - 1) * (300 / 19)} 
            y="170" 
            textAnchor="middle"
            fill={colors.axis}
            fontSize="10"
          >
            {num}
          </text>
        ))}
        
        {/* X-axis title */}
        <text 
          x="150" 
          y="180" 
          textAnchor="middle"
          fill={colors.axis}
          fontSize="11"
        >
          Neuron Threshold
        </text>
        
        {/* Y-axis labels */}
        <text 
          x="-75" 
          y="10" 
          textAnchor="middle"
          fill={colors.axis}
          fontSize="10"
          transform="rotate(-90)"
        >
          {type.charAt(0).toUpperCase() + type.slice(1)}
        </text>
      </svg>
    </div>
  );
};

export default Chart;
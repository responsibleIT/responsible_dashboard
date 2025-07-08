import React, { useState, useEffect, useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Label
} from 'recharts';

/**
 * Threshold Chart component using Recharts with support for multiple metrics
 * @param {string} type - Chart type (power, accuracy, carbon, computing)
 * @param {string} metric - Performance metric to display (accuracy, f1, precision, recall)
 * @param {number} threshold - Current neuron threshold value
 * @param {boolean} refreshCharts - Whether to regenerate chart data when threshold changes
 * @param {boolean} includeZero - Whether to include a value for threshold = 0
 * @param {number} maxThreshold - Maximum threshold value (default: 5)
 * @param {Array} data - Array of data points from CSV
 */
const ThresholdChart = ({ 
  type, 
  metric = 'accuracy', 
  threshold, 
  refreshCharts = false, 
  includeZero = false, 
  maxThreshold = 3, 
  data 
}) => {
  // Chart configuration based on type
  const chartConfig = {
    power: {
      title: 'Power Usage vs Threshold',
      dataKey: 'power',
      yAxisLabel: 'Power (kWh)',
      lineColor: '#8884d8',
      tooltipFormatter: (value) => [`${value} kWh`, 'Power'],
      domain: [0, 'auto'],
      baseValue: 0.45
    },
    accuracy: {
      title: `${metric.charAt(0).toUpperCase() + metric.slice(1)} vs Threshold`,
      dataKey: metric,
      yAxisLabel: metric.charAt(0).toUpperCase() + metric.slice(1),
      lineColor: '#82ca9d',
      tooltipFormatter: (value) => [
        metric === 'accuracy' ? `${value.toFixed(1)}%` : value.toFixed(3), 
        metric.charAt(0).toUpperCase() + metric.slice(1)
      ],
      domain: metric === 'accuracy' ? [0, 100] : [0, 1],
      baseValue: metric === 'accuracy' ? 84.2 : 0.85
    },
    carbon: {
      title: 'Carbon Emissions vs Threshold',
      dataKey: 'carbonEmissions',
      yAxisLabel: 'Carbon (gCO2)',
      lineColor: '#ff8042',
      tooltipFormatter: (value) => [`${value} gCO2`, 'Emissions'],
      domain: [0, 'auto'],
      baseValue: 80
    },
    computing: {
      title: 'Computing Power vs Threshold',
      dataKey: 'computingPower',
      yAxisLabel: 'Computing (TFLOPS)',
      lineColor: '#d88884',
      tooltipFormatter: (value) => [`${value} TFLOPS`, 'Computing Power'],
      domain: [0, 12000],
      baseValue: 0.01
    }
  };

  const config = chartConfig[type] || chartConfig.power;

  // Process the data based on chart type and selected metric
  const chartData = useMemo(() => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return [];
    }
    
    // Sort data by threshold to ensure proper line rendering
    const sortedData = [...data].sort((a, b) => a.threshold - b.threshold);
    
    // Map the data according to chart type
    switch (type) {
      case 'accuracy':
        return sortedData.map(entry => ({
          threshold: entry.threshold,
          // Use dynamic metric based on selection
          [metric]: metric === 'accuracy' 
            ? entry[`${metric}_pruned`] * 100 
            : Number(entry[`${metric}_pruned`])
        }));
        
      case 'power':
        return sortedData.map(entry => ({
          threshold: entry.threshold,
          power: (entry.power * 1000).toFixed(4) // per 1000 calls
        }));
        
      case 'carbon':
        return sortedData.map(entry => ({
          threshold: entry.threshold,
          carbonEmissions: entry.carbonEmissions.toFixed(4) // per 1000 calls
        }));
        
      case 'computing':
        return sortedData.map(entry => ({
          threshold: entry.threshold,
          computingPower: (entry.computingPower / 1e9 * 1000).toFixed(4) // per 1000 calls
        }));
        
      default:
        return sortedData;
    }
  }, [data, type, metric]);

  // If no data is available, show a message
  if (chartData.length === 0) {
    return (
      <div className="d-flex justify-content-center align-items-center h-100">
        <span>No data available</span>
      </div>
    );
  }

  const numericThreshold = parseFloat(threshold);

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart 
        data={chartData}
        margin={{ top: 20, right: 30, left: 20, bottom: 25 }}
        background={{ fill: '#FDFDF7' }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis 
          dataKey="threshold" 
          fontSize={9}
          tickSize={5}
          domain={[0, maxThreshold]}
          ticks={[0, 1, 2, 3, 4, 5]} // Adjusted ticks for 0-5 range
        >
          <Label 
            value="Neuron Threshold" 
            position="insideBottom" 
            offset={-10} 
            fontSize={9}
            style={{ textAnchor: 'middle' }}
          />
        </XAxis>
        <YAxis 
          domain={config.domain}
          fontSize={9}
          width={45}
          tickFormatter={(value) => value}
        >
          <Label 
            value={config.yAxisLabel}
            angle={-90}
            position="insideLeft"
            offset={-5}
            fontSize={9}
            style={{ textAnchor: 'middle' }}
          />
        </YAxis>
        <Tooltip 
          formatter={config.tooltipFormatter}
          labelFormatter={(value) => `Threshold: ${value}`}
        />
        <Line 
          type="monotone" 
          dataKey={config.dataKey} 
          stroke={config.lineColor} 
          activeDot={{ r: 6 }} 
          strokeWidth={2}
          dot={false} // Hide individual dots for smoother appearance
          isAnimationActive={false} // Disable animation for better performance
        />
        {!isNaN(numericThreshold) && (
          <ReferenceLine 
            x={numericThreshold} 
            stroke="red" 
            strokeWidth={2}
          >
            <Label
              value={numericThreshold.toFixed(1)}
              position="top"
              fill="red"
              fontSize={9}
              offset={10}
            />
          </ReferenceLine>
        )}
      </LineChart>
    </ResponsiveContainer>
  );
};

export default ThresholdChart;
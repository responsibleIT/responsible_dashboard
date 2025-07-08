/**
 * Utility functions for the application
 */

/**
 * Validates a Hugging Face URL format
 * @param {string} url - URL to validate
 * @returns {boolean} - True if valid HF URL
 */
export const isValidHuggingFaceUrl = (url) => {
    if (!url) return false;
    
    // Simple validation - could be more sophisticated in a real app
    return url.startsWith('https://huggingface.co/');
  };
  
  /**
   * Formats numbers with proper units
   * @param {number} value - The number to format
   * @param {string} unit - The unit to append
   * @returns {string} - Formatted string with value and unit
   */
  export const formatWithUnit = (value, unit) => {
    return `${value} ${unit}`;
  };
  
  /**
   * Maps threshold values to chart points for visualization
   * @param {number} threshold - Current threshold value
   * @param {string} metricType - Type of metric (power, accuracy, etc.)
   * @returns {Array} - Array of data points for the chart
   */
  export const getChartPoints = (threshold, metricType) => {
    // This would normally calculate actual data points based on the model
    // For now, we return mock data points
    const mockData = {
      power: [0.4, 0.38, 0.36, 0.35, 0.33, 0.32, 0.31, 0.3, 0.29, 0.28],
      accuracy: [95, 92, 90, 87, 84, 80, 76, 73, 70, 67],
      carbon: [75, 70, 65, 62, 58, 55, 52, 50, 48, 45],
      computing: [32, 30, 28, 26, 24, 22, 21, 20, 19, 18]
    };
    
    return mockData[metricType] || [];
  };
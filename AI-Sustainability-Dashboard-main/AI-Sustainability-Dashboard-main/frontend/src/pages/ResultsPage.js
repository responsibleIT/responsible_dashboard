import React, { useState, useEffect } from 'react';
import { Row, Col, Button, Form, Card, Nav } from 'react-bootstrap';
import ThresholdChart from '../components/charts/ThresholdChart';
import Papa from 'papaparse';
import { benchmarkData } from '../data/benchmark';

/**
 * Page component for displaying optimization results
 * Fixed heights and no scrolling
 * With consistent background color
 */
const ResultsPage = ({ modelData, handleBack, handleValidate }) => {
  const [threshold, setThreshold] = useState(0);
  const [gpuModel, setGpuModel] = useState("NVIDIA A100");
  const [location, setLocation] = useState("France");
  const [performanceMetric, setPerformanceMetric] = useState("Accuracy");
  const [activeTab, setActiveTab] = useState("charts");
  const [csvData, setCsvData] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  
  // Class performance data (will be updated dynamically)
  const [classPerformanceData, setClassPerformanceData] = useState([
    { class: "Positive", original: 86.2, pruned: 86.2 },
    { class: "Neutral", original: 83.7, pruned: 83.7 },
    { class: "Negative", original: 82.5, pruned: 82.5 }
  ]);

  const onValidateClick = () => {
    handleValidate(threshold);
  };

  // Mock CSV data to use when file loading fails
  const getMockData = () => {
    return benchmarkData;
  };

  // Load CSV data
  useEffect(() => {
    const loadCSVData = async () => {
      try {
        setIsLoading(true);
        
        let transformedData;
        
        // Try to load from file if window.fs is available
        if (window.fs && typeof window.fs.readFile === 'function') {
          // Replace 'threshold_data.csv' with your actual CSV file path
          const fileContent = await window.fs.readFile('threshold_data.csv', { encoding: 'utf8' });
          
          // Parse CSV
          const parseResult = Papa.parse(fileContent, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true
          });
          
          if (parseResult.errors.length > 0) {
            console.error("CSV parsing errors:", parseResult.errors);
          }
          
          transformedData = parseResult.data;
        } else {
          // Fallback to mock data if window.fs is not available
          console.log("window.fs.readFile not available, using mock data instead");
          transformedData = getMockData();
        }
        
        // Transform data to match our data structure
        let firstRow = transformedData[0];
        
        let hardwarePerformance = 17.4; // 17.4 TFLOPS
        let hardwarePower = 400; // 400W
        let efficiency = hardwarePerformance * 1e9 / hardwarePower; // 43.5 TFLOPS/W


        const processedData = transformedData.map(row => {
          
          const flops = row.flops;
          const energy = flops / efficiency; // Energy in Joules
          const power = energy / 3600000
          const carbonEmissionPerKWh = 50; // gCO2/kWh 
          
          return {
            threshold: row.threshold.toFixed(1),

            power_baseline: firstRow.flops / efficiency / 3600000,
            power: power,

            carbonEmmissions_baseline: firstRow.flops / efficiency / 3600000 * carbonEmissionPerKWh,
            carbonEmissions: power * carbonEmissionPerKWh,

            computingPower_baseline: firstRow.flops,
            computingPower: row.flops,

            accuracy_baseline: firstRow.overall_accuracy,
            accuracy_negative_baseline: firstRow.negative_accuracy,
            accuracy_neutral_baseline: firstRow.neutral_accuracy,
            accuracy_positive_baseline: firstRow.positive_accuracy,
            f1_baseline: firstRow.overall_f1,
            f1_negative_baseline: firstRow.negative_f1,
            f1_neutral_baseline: firstRow.neutral_f1,
            f1_positive_baseline: firstRow.positive_f1,
            precision_baseline: firstRow.overall_precision,
            precision_negative_baseline: firstRow.negative_precision,
            precision_neutral_baseline: firstRow.neutral_precision,
            precision_positive_baseline: firstRow.positive_precision,
            recall_baseline: firstRow.overall_recall,
            recall_negative_baseline: firstRow.negative_recall,
            recall_neutral_baseline: firstRow.neutral_recall,
            recall_positive_baseline: firstRow.positive_recall,

            accuracy_pruned: row.overall_accuracy,
            accuracy_negative_pruned: row.negative_accuracy,
            accuracy_neutral_pruned: row.neutral_accuracy,
            accuracy_positive_pruned: row.positive_accuracy,
            f1_pruned: row.overall_f1,
            f1_negative_pruned: row.negative_f1,
            f1_neutral_pruned: row.neutral_f1,
            f1_positive_pruned: row.positive_f1,
            precision_pruned: row.overall_precision,
            precision_negative_pruned: row.negative_precision,
            precision_neutral_pruned: row.neutral_precision,
            precision_positive_pruned: row.positive_precision,
            recall_pruned: row.overall_recall,
            recall_negative_pruned: row.negative_recall,
            recall_neutral_pruned: row.neutral_recall,
            recall_positive_pruned: row.positive_recall,
          }
        });
        
        setCsvData(processedData);

        setMetrics({
          power: processedData[0].power_baseline,
          accuracy: processedData[0].accuracy_baseline,
          carbonFootprint: processedData[0].carbonEmmissions_baseline,
          computingPower: processedData[0].computingPower_baseline
        });
        
        if (processedData.length > 0) {
          updateClassPerformanceData(processedData[0], performanceMetric);
        }
        
        setIsLoading(false);
      } catch (error) {
        console.error("Error loading data:", error);
        
        setIsLoading(false);
      }
    };
    
    loadCSVData();
  }, []);

  // Helper function to update class performance data based on metric
  const updateClassPerformanceData = (dataEntry, metric) => {
    if (!dataEntry) return;
    
    // All values should be on a 0-100 scale for the progress bars
    setClassPerformanceData([
      { 
        class: "Positive", 
        original: metric === 'Accuracy' 
          ? dataEntry[`${metric}_positive_baseline`] * 100
          : dataEntry[`${metric}_positive_baseline`], 
        pruned: metric === 'Accuracy' 
          ? dataEntry[`${metric}_positive_pruned`] * 100
          : dataEntry[`${metric}_positive_pruned`] 
      },
      { 
        class: "Neutral", 
        original: metric === 'Accuracy' 
          ? dataEntry[`${metric}_neutral_baseline`] * 100
          : dataEntry[`${metric}_neutral_baseline`], 
        pruned: metric === 'Accuracy' 
          ? dataEntry[`${metric}_neutral_pruned`] * 100
          : dataEntry[`${metric}_neutral_pruned`] 
      },
      { 
        class: "Negative", 
        original: metric === 'Accuracy' 
          ? dataEntry[`${metric}_negative_baseline`] * 100
          : dataEntry[`${metric}_negative_baseline`], 
        pruned: metric === 'Accuracy' 
          ? dataEntry[`${metric}_negative_pruned`] * 100
          : dataEntry[`${metric}_negative_pruned`] 
      }
    ]);
  };
  
  // Metric values to update based on threshold
  const [metrics, setMetrics] = useState({
    power: csvData.length > 0 ? csvData[0].power_baseline : 0,
    accuracy: csvData.length > 0 ? csvData[0].accuracy_baseline : 0,
    carbonFootprint: csvData.length > 0 ? csvData[0].carbonEmmissions_baseline : 0,
    computingPower: csvData.length > 0 ? csvData[0].computingPower : 0
  });

  // Percentage changes from base values
  const [percentChanges, setPercentChanges] = useState({
    power: 0,
    accuracy: 0,
    carbonFootprint: 0,
    computingPower: 0
  });

  // Handle slider change - convert from string to float with 1 decimal place
  const handleSliderChange = (e) => {
    const value = parseFloat(parseFloat(e.target.value).toFixed(1));
    setThreshold(value);
  };

  // Function to calculate metrics based on threshold
  useEffect(() => {
    if (csvData.length === 0) return;
    
    // Find the closest threshold value in our data
    let entry = csvData.find(d => d.threshold === parseFloat(threshold.toFixed(1)));
    
    if (!entry) {
      // If the exact threshold is not found, find the closest one
      const sortedData = [...csvData].sort((a, b) => 
        Math.abs(a.threshold - threshold) - Math.abs(b.threshold - threshold)
      );
      entry = sortedData[0];
    }
    
    if (!entry) return;
  
    // Calculate percentage changes from base values
    const powerChange = ((entry.power - entry.power_baseline) / entry.power_baseline * 100).toFixed(1);
    const accuracyChange = ((entry.accuracy_pruned - entry.accuracy_baseline) / entry.accuracy_baseline * 100).toFixed(1);
    const carbonChange = ((entry.carbonEmissions - entry.carbonEmmissions_baseline) / entry.carbonEmmissions_baseline * 100).toFixed(1);
    const computingChange = ((entry.computingPower - entry.computingPower_baseline) / entry.computingPower_baseline * 100).toFixed(1);

    setMetrics({
      power: entry.power,
      accuracy: entry.accuracy_pruned,
      carbonFootprint: entry.carbonEmissions,
      computingPower: entry.computingPower
    });
  
    setPercentChanges({
      power: Number(powerChange),
      accuracy: Number(accuracyChange),
      carbonFootprint: Number(carbonChange),
      computingPower: Number(computingChange)
    });
    
    // Update the class performance data based on threshold
    setClassPerformanceData([
      { class: "Positive", original: entry.accuracy_positive_baseline, pruned: entry.accuracy_positive_pruned },
      { class: "Neutral", original: entry.accuracy_neutral_baseline, pruned: entry.accuracy_neutral_pruned },
      { class: "Negative", original: entry.accuracy_negative_baseline, pruned: entry.accuracy_negative_pruned }
    ]);
  }, [threshold]);

  // Helper to render percentage change indicators
  const renderChangeIndicator = (change, type) => {
    let isGoodChange = change < 0;
    const arrow = change >= 0 ? '↑' : '↓';
    
    if (type === 'accuracy') {
      isGoodChange = change >= 0;
    }

    const color = isGoodChange ? 'green' : 'red';
    const displayValue = Math.abs(change);
    
    return (
      <span className="ms-2" style={{ color, fontSize: '0.85rem', fontWeight: 'bold' }}>
        {arrow} {displayValue}%
      </span>
    );
  };

  // GPU model options
  const gpuOptions = [
    "NVIDIA A100",
    "NVIDIA V100",
    "NVIDIA T4",
    "AMD MI100",
    "Google TPU v4"
  ];

  // Location options
  const locationOptions = [
    "France",
    "United States",
    "Germany",
    "Singapore",
    "Australia"
  ];

  // Performance metric options
  const performanceMetricOptions = [
    "Accuracy",
    "F1 Score",
    "Recall",
    "Precision",
    "Throughput"
  ];

  // Render the Charts view
  const renderChartsView = () => (
    <div className="flex-grow-1 mx-0 mb-3 d-flex flex-column" style={{ minHeight: 0 }}>
      <div className="row mx-0 flex-grow-1 mb-2">
        <div className="col-md-6 ps-2 pe-2 h-100">
          <Card className="border rounded-1 shadow-sm h-100" style={{ backgroundColor: '#FDFDF7', minHeight: '250px' }}>
            <Card.Body className="p-2 d-flex flex-column">
              <h6 className="fw-bold mb-1">Power Usage (per 1000 calls)</h6>
              <div className="flex-grow-1" style={{ minHeight: '220px' }}>
                <ThresholdChart type="power" threshold={threshold} refreshCharts={false} includeZero={true} maxThreshold={10} data={csvData} />
              </div>
            </Card.Body>
          </Card>
        </div>
        <div className="col-md-6 ps-2 pe-2 h-100">
          <Card className="border rounded-1 shadow-sm h-100" style={{ backgroundColor: '#FDFDF7', minHeight: '250px' }}>
            <Card.Body className="p-2 d-flex flex-column">
              <h6 className="fw-bold mb-1">Predicted Model Accuracy</h6>
              <div className="flex-grow-1" style={{ minHeight: '220px' }}>
                <ThresholdChart type="accuracy" threshold={threshold} refreshCharts={false} includeZero={true} maxThreshold={10} data={csvData} />
              </div>
            </Card.Body>
          </Card>
        </div>
      </div>
      
      <div className="row mx-0 flex-grow-1" style={{ height: 'calc(50% - 5px)' }}>
        <div className="col-md-6 ps-2 pe-2 h-100">
          <Card className="border rounded-1 shadow-sm h-100" style={{ backgroundColor: '#FDFDF7', minHeight: '250px' }}>
            <Card.Body className="p-2 d-flex flex-column">
              <h6 className="fw-bold mb-1">Carbon Emissions (per 1000 calls)</h6>
              <div className="flex-grow-1" style={{ minHeight: '220px' }}>
                <ThresholdChart type="carbon" threshold={threshold} refreshCharts={false} includeZero={true} maxThreshold={10} data={csvData} />
              </div>
            </Card.Body>
          </Card>
        </div>
        <div className="col-md-6 ps-2 pe-2 h-100">
          <Card className="border rounded-1 shadow-sm h-100" style={{ backgroundColor: '#FDFDF7', minHeight: '250px' }}>
            <Card.Body className="p-2 d-flex flex-column">
              <h6 className="fw-bold mb-1">Computing Power</h6>
              <div className="flex-grow-1" style={{ minHeight: '220px' }}>
                <ThresholdChart type="computing" threshold={threshold} refreshCharts={false} includeZero={true} maxThreshold={10} data={csvData} />
              </div>
            </Card.Body>
          </Card>
        </div>
      </div>
    </div>
  );

  const renderPerformancePerClassView = () => (
    <div className="mx-0 mb-3" style={{ overflowY: 'auto' }}>
      <div className="px-2">
        {isLoading ? (
          <div className="text-center my-4">
            <span>Loading performance data...</span>
          </div>
        ) : (
          classPerformanceData.map((item, index) => (
            <div key={index} className="mb-4">
              <h6 className="mb-2">{item.class}</h6>
              <div className="mb-2">
                <div className="d-flex justify-content-between mb-1">
                  <span style={{ fontSize: '14px' }}>Original</span>
                  <span style={{ fontSize: '14px' }}>
                    {performanceMetric === 'Accuracy' 
                      ? `${(item.original * 100).toFixed(1)}%` 
                      : item.original.toFixed(1)
                    }
                  </span>
                </div>
                <div className="progress" style={{ height: '10px' }}>
                  <div 
                    className="progress-bar bg-primary" 
                    role="progressbar" 
                    style={{ width: `${item.original * 100}%` }} 
                    aria-valuenow={item.original} 
                    aria-valuemin="0" 
                    aria-valuemax="100"
                  ></div>
                </div>
              </div>
              <div>
                <div className="d-flex justify-content-between mb-1">
                  <span style={{ fontSize: '14px' }}>Pruned</span>
                  <span style={{ fontSize: '14px' }}>
                    {performanceMetric === 'Accuracy' 
                      ? `${(item.pruned * 100).toFixed(1)}%` 
                      : item.pruned.toFixed(1)
                    }
                  </span>
                </div>
                <div className="progress" style={{ height: '10px' }}>
                  <div 
                    className="progress-bar bg-success" 
                    role="progressbar" 
                    style={{ width: `${item.pruned * 100}%` }} 
                    aria-valuenow={item.pruned} 
                    aria-valuemin="0" 
                    aria-valuemax="100"
                  ></div>
                </div>
              </div>
              
              {/* Show the change percentage */}
              <div className="mt-1 text-end">
                {renderChangeIndicator(
                  ((item.pruned - item.original) / item.original * 100).toFixed(1), 
                  'accuracy'
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  return (
    <div className="d-flex flex-column" style={{ height: '100vh', backgroundColor: '#FDFDF7', overflow: 'hidden' }}>
      {/* Header with exactly 60px height */}
      <div className="border-bottom d-flex align-items-center" style={{ height: '60px', backgroundColor: '#FDFDF7', position: 'sticky', top: 0, zIndex: 10 }}>
        <div className="d-flex justify-content-between align-items-center w-100 px-3">
          <Button 
            variant="outline-dark" 
            onClick={handleBack}
            className="px-3 rounded-1"
            size="sm"
          >
            Back
          </Button>
          
          <h5 className="mb-0 fw-medium">cardiffnlp / twitter-xlm-roberta-base-sentiment</h5>
          
          <Button 
            variant="dark" 
            onClick={onValidateClick}
            className="px-3 rounded-1"
            size="sm"
          >
            Validate
          </Button>
        </div>
      </div>
      
      {/* Main content area */}
      <div className="d-flex flex-column" style={{ height: 'calc(100vh - 60px)', backgroundColor: '#FDFDF7', padding: '0', flex: '1 1 auto', overflow: 'auto' }}>
        {/* Model Information and Threshold */}
        <Row className="mx-0 mt-3 mb-3">
          <Col md={3} className="ps-2 pe-2">
            <Form.Group>
              <Form.Label className="mb-1 small">GPU model:</Form.Label>
              <Form.Select 
                value={gpuModel}
                onChange={(e) => setGpuModel(e.target.value)}
                className="form-control-sm"
                style={{ backgroundColor: '#FDFDF7' }}
              >
                {gpuOptions.map((option, index) => (
                  <option key={index} value={option}>{option}</option>
                ))}
              </Form.Select>
            </Form.Group>
          </Col>
          <Col md={3} className="ps-2 pe-2">
            <Form.Group>
              <Form.Label className="mb-1 small">Location:</Form.Label>
              <Form.Select
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="form-control-sm"
                style={{ backgroundColor: '#FDFDF7' }}
              >
                {locationOptions.map((option, index) => (
                  <option key={index} value={option}>{option}</option>
                ))}
              </Form.Select>
            </Form.Group>
          </Col>
          <Col md={3} className="ps-2 pe-2">
            <Form.Group>
              <Form.Label className="mb-1 small">Performance Metric:</Form.Label>
              <Form.Select
                value={performanceMetric}
                onChange={(e) => setPerformanceMetric(e.target.value)}
                className="form-control-sm"
                style={{ backgroundColor: '#FDFDF7' }}
              >
                {performanceMetricOptions.map((option, index) => (
                  <option key={index} value={option}>{option}</option>
                ))}
              </Form.Select>
            </Form.Group>
          </Col>
          <Col md={3} className="ps-2 pe-2">
            <Form.Group>
              <Form.Label className="mb-1 small">Neuron Threshold: {threshold}</Form.Label>
              <div className="d-flex align-items-center">
                <div className="w-100">
                  <Form.Range 
                    min="0" 
                    max="3" 
                    step="0.1"
                    value={threshold}
                    onChange={handleSliderChange}
                    className="form-range"
                  />
                </div>
              </div>
            </Form.Group>
          </Col>
        </Row>

        {/* Disclaimer at bottom */}
        <div className="mx-2 mb-3">
          <div className="border border-warning bg-warning bg-opacity-10 rounded-0 p-2">
            <p className="small mb-0">
              <strong>Disclaimer:</strong> The performance metric shown below is a prediction and may differ from actual measurements. To export the model, an actual benchmark on the provided validation set must be performed with the selected threshold. Click the "Validate" button at the top right corner to start this benchmark.
            </p>
          </div>
        </div>
        
        {/* Metrics Cards - Fixed height 69px */}
        <Row className="mx-0 mb-3">
          <Col md={3} className="ps-2 pe-2">
            <Card className="border-0 rounded-3" style={{ backgroundColor: '#f3e5ff', height: '69px' }}>
              <Card.Body className="p-2 d-flex flex-column justify-content-between">
                <Card.Title className="small mb-0">Power (per 1000 calls)</Card.Title>
                <div className="d-flex align-items-center">
                  <h3 className="mb-0 fw-bold" style={{ color: '#6f42c1' }}>{(metrics.power * 1000).toFixed(4)} kWh</h3>
                  {renderChangeIndicator(percentChanges.power, 'power')}
                </div>
              </Card.Body>
            </Card>
          </Col>
          <Col md={3} className="ps-2 pe-2">
            <Card className="border-0 rounded-3" style={{ backgroundColor: '#e5ffe5', height: '69px' }}>
              <Card.Body className="p-2 d-flex flex-column justify-content-between">
                <Card.Title className="small mb-0">Predicted Accuracy</Card.Title>
                <div className="d-flex align-items-center">
                  <h3 className="mb-0 fw-bold" style={{ color: '#198754' }}>{(metrics.accuracy * 100.0).toFixed(1)}%</h3>
                  {renderChangeIndicator(percentChanges.accuracy, 'accuracy')}
                </div>
              </Card.Body>
            </Card>
          </Col>
          <Col md={3} className="ps-2 pe-2">
            <Card className="border-0 rounded-3" style={{ backgroundColor: '#fff5e5', height: '69px' }}>
              <Card.Body className="p-2 d-flex flex-column justify-content-between">
                <Card.Title className="small mb-0">Carbon Footprint (per 1000 calls)</Card.Title>
                <div className="d-flex align-items-center">
                  <h3 className="mb-0 fw-bold" style={{ color: '#fd7e14' }}>{(metrics.carbonFootprint * 1000).toFixed(4)} gCO2</h3>
                  {renderChangeIndicator(percentChanges.carbonFootprint, 'carbon')}
                </div>
              </Card.Body>
            </Card>
          </Col>
          <Col md={3} className="ps-2 pe-2">
            <Card className="border-0 rounded-3" style={{ backgroundColor: '#ffe5e5', height: '69px' }}>
              <Card.Body className="p-2 d-flex flex-column justify-content-between">
                <Card.Title className="small mb-0">Computing Power</Card.Title>
                <div className="d-flex align-items-center">
                  <h3 className="mb-0 fw-bold" style={{ color: '#dc3545' }}>{(metrics.computingPower / 1e9).toFixed(4)} TFLOPS</h3>
                  {renderChangeIndicator(percentChanges.computingPower, 'computing')}
                </div>
              </Card.Body>
            </Card>
          </Col>
        </Row>
        
        {/* Tab Navigation */}
        <div className="mx-2 mb-3 border-bottom">
          <Nav
            variant="tabs"
            activeKey={activeTab}
            onSelect={(k) => setActiveTab(k)}
            className="border-0"
          >
            <Nav.Item>
              <Nav.Link 
                eventKey="charts" 
                className={`border-0 rounded-0 px-4 ${activeTab === 'charts' ? 'text-primary border-primary border-bottom border-3' : 'text-secondary'}`}
                style={{ backgroundColor: 'transparent' }}
              >
                Charts
              </Nav.Link>
            </Nav.Item>
            <Nav.Item>
              <Nav.Link 
                eventKey="performance" 
                className={`border-0 rounded-0 px-4 ${activeTab === 'performance' ? 'text-primary border-primary border-bottom border-3' : 'text-secondary'}`}
                style={{ backgroundColor: 'transparent' }}
              >
                Performance per Class
              </Nav.Link>
            </Nav.Item>
          </Nav>
        </div>
        
        {/* Charts that fill remaining space */}
        {activeTab === 'charts' ? renderChartsView() : renderPerformancePerClassView()}
      </div>
    </div>
  );
};

export default ResultsPage;
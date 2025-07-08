import React, { useState } from 'react';
import { Container, Row, Col, Button, Card, ProgressBar, Accordion } from 'react-bootstrap';
import { ChevronDown, ChevronUp } from 'react-bootstrap-icons'; // For expand/collapse icons

// Helper to render percentage change indicators (can be imported or defined here)
const renderChangeIndicator = (change, type) => {
    let isGoodChange = change < 0; // Default: lower is better
    const arrow = change > 0 ? '↑' : change < 0 ? '↓' : '';
    const displayValue = Math.abs(change);

    if (type === 'accuracy' || type === 'f1' || type === 'precision' || type === 'recall') {
        isGoodChange = change >= 0; // Higher is better for these metrics
    }

    const color = change === 0 ? 'grey' : isGoodChange ? 'green' : 'red';

    if (change === 0) {
        return <span className="ms-1" style={{ color, fontSize: '0.8rem', fontWeight: 'normal' }}>(- 0.0%)</span>;
    }

    return (
        <span className="ms-1" style={{ color, fontSize: '0.8rem', fontWeight: 'bold' }}>
            ({arrow} {displayValue}%)
        </span>
    );
};

// Helper to calculate percentage change
const calculateChange = (current, original) => {
    if (original === null || original === undefined || original === 0 || current === null || current === undefined) return 0;
    return parseFloat(((current - original) / original * 100).toFixed(1));
};

// Helper Component for Metric Progress Bars
const MetricProgressBar = ({ label, original, pruned, type }) => {
    const originalValue = parseFloat(original) || 0;
    const prunedValue = parseFloat(pruned) || 0;
    const change = calculateChange(prunedValue, originalValue);

    // Convert 0-1 scale to 0-100 for progress bars and display
    const displayOriginal = (originalValue * 100).toFixed(1);
    const displayPruned = (prunedValue * 100).toFixed(1);


    return (
        <div className="mb-3">
            <div className='small mb-1 fw-medium'>{label}</div>
            {/* Original */}
            <div className="d-flex justify-content-between align-items-center mb-1">
                <span style={{ fontSize: '13px', color: '#6c757d' }}>Original</span>
                <span style={{ fontSize: '13px', color: '#6c757d', fontWeight: '500' }}>{displayOriginal}%</span>
            </div>
            <ProgressBar now={displayOriginal} style={{ height: '8px' }} variant="secondary" />
            {/* Pruned */}
            <div className="d-flex justify-content-between align-items-center mb-1 mt-2">
                 <span style={{ fontSize: '13px' }}>Pruned</span>
                 <div style={{ fontSize: '13px', fontWeight: '500' }}>
                      {displayPruned}%
                      {renderChangeIndicator(change, type)}
                  </div>
            </div>
            <ProgressBar now={displayPruned} style={{ height: '8px' }} variant="success" />
        </div>
    );
};


const BenchmarkResultsPage = ({ modelName, validatedData, baselineData, originalParams, gpuModel, location, onBack, onExport }) => {

  const [openAccordionKey, setOpenAccordionKey] = useState(null); // Track open accordion item

  if (!validatedData || !baselineData) {
    return <div>Error: Missing validation data. <Button variant="link" onClick={onBack}>Go Back</Button></div>;
  }

  // --- Calculate Metrics ---
  const prunedParams = originalParams * (1 - (validatedData.params_reduction_pct || 0) / 100);
  const paramsReduction = validatedData.params_reduction_pct || 0;

  console.log('Validated Data:', validatedData);
  console.log('Baseline Data:', baselineData);

  const metrics = {
      accuracy: { original: baselineData.overall_accuracy, pruned: validatedData.overall_accuracy },
      power: { original: baselineData.power * 1000, pruned: validatedData.power * 1000 },
      carbon: { original: baselineData.carbonEmissions * 1000, pruned: validatedData.carbonEmissions * 1000 },
      compute: { original: baselineData.flops / 1e12, pruned: validatedData.flops / 1e12 },
  };

  const classes = ["Negative", "Neutral", "Positive"];
  const classMetrics = {};
  classes.forEach(cls => {
      const lowerCls = cls.toLowerCase();
      classMetrics[cls] = {
          accuracy: { original: baselineData[`${lowerCls}_accuracy_baseline`], pruned: validatedData[`${lowerCls}_accuracy_pruned`] },
          f1: { original: baselineData[`${lowerCls}_f1_baseline`], pruned: validatedData[`${lowerCls}_f1_pruned`] },
          precision: { original: baselineData[`${lowerCls}_precision_baseline`], pruned: validatedData[`${lowerCls}_precision_pruned`] },
          recall: { original: baselineData[`${lowerCls}_recall_baseline`], pruned: validatedData[`${lowerCls}_recall_pruned`] }
      };
  });


  const formatNumber = (num, decimals = 0) => {
    if (num === null || num === undefined) return 'N/A';
    return num.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };


  return (
    <div className="d-flex flex-column" style={{ height: '100vh', backgroundColor: '#FDFDF7', overflow: 'hidden' }}>
      {/* Header */}
       <div className="border-bottom d-flex align-items-center" style={{ height: '60px', backgroundColor: '#FDFDF7', position: 'sticky', top: 0, zIndex: 10 }}>
           <div className="d-flex justify-content-between align-items-center w-100 px-3">
               <Button variant="outline-dark" onClick={onBack} className="px-3 rounded-1" size="sm">Back</Button>
               <h5 className="mb-0 fw-medium">{modelName}</h5>
               <Button variant="dark" onClick={onExport} className="px-3 rounded-1" size="sm">Export model</Button>
           </div>
       </div>

      {/* Main Content Scrollable */}
      <div style={{ height: 'calc(100vh - 60px)', overflowY: 'auto', padding: '1rem 1.5rem' }}>
          {/* Benchmark Settings */}
          <Card className="mb-4 shadow-sm border rounded-1">
              <Card.Header style={{backgroundColor: '#f8f9fa'}}>Benchmark settings</Card.Header>
              <Card.Body>
                  <Row>
                      <Col md={3}><small className="text-muted">Model</small><div>{modelName}</div></Col>
                      <Col md={3}><small className="text-muted">GPU</small><div>{gpuModel}</div></Col>
                      <Col md={3}><small className="text-muted">Location</small><div>{location}</div></Col>
                      <Col md={3}><small className="text-muted">Threshold</small><div>{validatedData.threshold.toFixed(1)}%</div></Col>
                  </Row>
              </Card.Body>
          </Card>

           {/* Model Parameters */}
           <Card className="mb-4 shadow-sm border rounded-1">
               <Card.Header style={{backgroundColor: '#f8f9fa'}}>Model parameters</Card.Header>
               <Card.Body>
                   <Row className="align-items-center">
                       <Col md={9}>
                           <small className="text-muted">Pruning reduced model size by {paramsReduction.toFixed(1)}%</small>
                           <ProgressBar style={{ height: '10px' }} className="mt-1">
                              {/* Show remaining portion */}
                               <ProgressBar striped variant="success" now={100 - paramsReduction} key={1} />
                               {/* Show pruned portion (optional visualization) */}
                               {/* <ProgressBar striped variant="danger" now={paramsReduction} key={2} /> */}
                           </ProgressBar>
                       </Col>
                       <Col md={3}>
                            <div className="d-grid" style={{ 
                                gridTemplateColumns: "auto auto auto", 
                                justifyContent: "end",
                                gap: "0.5rem"
                            }}>
                                <span className="text-muted small text-end">Original</span>
                                <span className="fw-bold text-end">{formatNumber(originalParams)}</span>
                                <span></span> {/* Empty placeholder for consistent grid */}
                                
                                <span className="text-muted small text-end">Pruned</span>
                                <span className="fw-bold text-end">{formatNumber(prunedParams)}</span>
                                <span style={{ color: 'red', fontSize: '0.8rem', fontWeight: 'bold' }}>
                                    ({paramsReduction > 0 ? `↓ ${paramsReduction.toFixed(1)}%` : '- 0.0%'})
                                </span>
                            </div>
                        </Col>
                   </Row>
               </Card.Body>
           </Card>


          {/* Overall Metrics */}
          <Row className="mb-4">
              {/* Accuracy */}
              <Col md={3}>
                  <Card className="h-100 shadow-sm border rounded-1" style={{ backgroundColor: '#e5ffe5' }}>
                       <Card.Body className="p-3">
                           <div className="d-flex justify-content-between align-items-start mb-2">
                               <span className="fw-medium">Accuracy</span>
                                {renderChangeIndicator(calculateChange(metrics.accuracy.pruned, metrics.accuracy.original), 'accuracy')}
                           </div>
                            <h4 className="fw-bold mb-1" style={{ color: '#198754' }}>{(metrics.accuracy.pruned * 100).toFixed(1)}%</h4>
                            <span className="text-muted small">Original: {(metrics.accuracy.original * 100).toFixed(1)}%</span>
                       </Card.Body>
                  </Card>
              </Col>
              {/* Power */}
              <Col md={3}>
                   <Card className="h-100 shadow-sm border rounded-1" style={{ backgroundColor: '#f3e5ff' }}>
                       <Card.Body className="p-3">
                           <div className="d-flex justify-content-between align-items-start mb-2">
                               <span className="fw-medium">Power <small>(per 1000 calls)</small></span>
                                {renderChangeIndicator(calculateChange(metrics.power.pruned, metrics.power.original), 'power')}
                           </div>
                            <h4 className="fw-bold mb-1" style={{ color: '#6f42c1' }}>{metrics.power.pruned.toFixed(3)} kWh</h4>
                            <span className="text-muted small">Original: {metrics.power.original.toFixed(3)}</span>
                       </Card.Body>
                   </Card>
              </Col>
              {/* Carbon Footprint */}
              <Col md={3}>
                    <Card className="h-100 shadow-sm border rounded-1" style={{ backgroundColor: '#fff5e5' }}>
                        <Card.Body className="p-3">
                           <div className="d-flex justify-content-between align-items-start mb-2">
                               <span className="fw-medium">Carbon <small>(per 1000 calls)</small></span>
                                {renderChangeIndicator(calculateChange(metrics.carbon.pruned, metrics.carbon.original), 'carbon')}
                           </div>
                            <h4 className="fw-bold mb-1" style={{ color: '#fd7e14' }}>{metrics.carbon.pruned.toFixed(3)} gCO2</h4>
                            <span className="text-muted small">Original: {metrics.carbon.original.toFixed(3)}</span>
                        </Card.Body>
                    </Card>
              </Col>
              {/* Computing Power */}
              <Col md={3}>
                   <Card className="h-100 shadow-sm border rounded-1" style={{ backgroundColor: '#ffe5e5' }}>
                       <Card.Body className="p-3">
                           <div className="d-flex justify-content-between align-items-start mb-2">
                               <span className="fw-medium">Compute <small>(TFLOPS)</small></span>
                                {renderChangeIndicator(calculateChange(metrics.compute.pruned, metrics.compute.original), 'compute')}
                           </div>
                           <h4 className="fw-bold mb-1" style={{ color: '#dc3545' }}>{metrics.compute.pruned.toFixed(4)} TFLOPS</h4>
                           <span className="text-muted small">Original: {metrics.compute.original.toFixed(4)}</span>
                       </Card.Body>
                   </Card>
              </Col>
          </Row>


          {/* Per-Class Metrics - Accordion */}
          <Accordion activeKey={openAccordionKey} onSelect={(key) => setOpenAccordionKey(key)} flush>
              {classes.map((className, index) => (
                  <Accordion.Item eventKey={String(index)} key={className} className="mb-2 border rounded-1 shadow-sm">
                       <Accordion.Header onClick={() => setOpenAccordionKey(openAccordionKey === String(index) ? null : String(index))}>
                           <div className="d-flex justify-content-between w-100 align-items-center pe-3">
                               <span>{className}</span>
                               <div className='small'>
                                    Accuracy: {(validatedData[`${className.toLowerCase()}_accuracy`] * 100).toFixed(1)}%
                                    {renderChangeIndicator(
                                        calculateChange(
                                            validatedData[`${className.toLowerCase()}_accuracy`], 
                                            baselineData[`${className.toLowerCase()}_accuracy`] || baselineData.overall_accuracy
                                        ), 
                                        'accuracy'
                                    )}
                                </div>
                           </div>
                       </Accordion.Header>
                      <Accordion.Body>
                          <Row>
                            <Col md={3}>
                                <MetricProgressBar 
                                    label="F1 Score" 
                                    original={baselineData[`${className.toLowerCase()}_f1`] || baselineData.overall_f1} 
                                    pruned={validatedData[`${className.toLowerCase()}_f1`]} 
                                    type="f1" 
                                />
                            </Col>
                            <Col md={3}>
                                <MetricProgressBar 
                                    label="Precision" 
                                    original={baselineData[`${className.toLowerCase()}_precision`] || baselineData.overall_precision} 
                                    pruned={validatedData[`${className.toLowerCase()}_precision`]} 
                                    type="precision" 
                                />
                            </Col>
                            <Col md={3}>
                                <MetricProgressBar 
                                    label="Recall" 
                                    original={baselineData[`${className.toLowerCase()}_recall`] || baselineData.overall_recall} 
                                    pruned={validatedData[`${className.toLowerCase()}_recall`]} 
                                    type="recall" 
                                />
                            </Col>
                            <Col md={3}>
                                <MetricProgressBar 
                                    label="Accuracy" 
                                    original={baselineData[`${className.toLowerCase()}_accuracy`] || baselineData.overall_accuracy} 
                                    pruned={validatedData[`${className.toLowerCase()}_accuracy`]} 
                                    type="accuracy" 
                                />
                            </Col>
                          </Row>
                      </Accordion.Body>
                  </Accordion.Item>
              ))}
          </Accordion>

      </div> {/* End Scrollable Area */}
    </div> // End Main Flex Container
  );
};

export default BenchmarkResultsPage;
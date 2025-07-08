import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Form, Button } from 'react-bootstrap';
import './InputPage.css'; // Import custom CSS for hover effects

/**
 * Page component for the Sustainability Dashboard
 * Matches the layout shown in the image with URL input, file upload options,
 * validation dataset section, and show optimizations button
 */
const InputPage = ({ modelUrl, setModelUrl, handleSubmit }) => {
  const [modelFile, setModelFile] = useState(null);
  const [validationDataset, setValidationDataset] = useState(null);
  const [isFormValid, setIsFormValid] = useState(false);
  
  // Check if both model and dataset are provided
  useEffect(() => {
    // Form is valid if either modelUrl OR modelFile is present, AND validationDataset is present
    const hasModel = modelUrl || modelFile;
    setIsFormValid(hasModel && validationDataset);
  }, [modelUrl, modelFile, validationDataset]);

  return (
    <Container className="d-flex justify-content-center align-items-center min-vh-100">
      <div className="border rounded p-4" style={{ width: '600px' }}>
        <h1 className="text-center mb-4">Sustainability Dashboard</h1>
        
        {/* Model Source Section */}
        <Form>
          <Form.Group className="mb-4">
            <Form.Label as="h4">Model source</Form.Label>
            
            {/* HuggingFace URL Input */}
            <Form.Label>Enter the HuggingFace URL of the model</Form.Label>
            <Form.Control 
              type="text" 
              value={modelUrl}
              onChange={(e) => {
                setModelUrl(e.target.value);
                if (e.target.value) setModelFile(null); // Clear file selection if URL is entered
              }}
              placeholder="https://huggingface.co/...."
              className="mb-2"
            />
            
            {/* Divider with "or" */}
            <div className="d-flex align-items-center my-3">
              <div className="flex-grow-1 border-bottom"></div>
              <div className="px-3">or</div>
              <div className="flex-grow-1 border-bottom"></div>
            </div>
            
            {/* Upload Model File */}
            <Form.Label>Upload model file</Form.Label>
            <Button 
              variant="outline-secondary" 
              className="custom-button w-100"
              onClick={() => document.getElementById('modelFileInput').click()}
            >
              Upload h5 model
            </Button>
            <Form.Control
              id="modelFileInput"
              type="file"
              onChange={(e) => {
                if (e.target.files[0]) {
                  setModelFile(e.target.files[0]);
                  setModelUrl(''); // Clear URL if file is selected
                }
              }}
              className="d-none"
            />
            {modelFile && (
              <div className="d-flex align-items-center mt-2 p-2 bg-light rounded">
                <span className="text-truncate flex-grow-1">{modelFile.name}</span>
                <button 
                  className="btn btn-sm text-danger" 
                  onClick={() => setModelFile(null)}
                  style={{ padding: '0 6px' }}
                  title="Remove file"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
                  </svg>
                </button>
              </div>
            )}
          </Form.Group>
          
                      {/* Validation Dataset Section */}
          <Form.Group className="mb-4">
            <Form.Label as="h4">Validation dataset</Form.Label>
            
            <Form.Label>Upload validation dataset (required)</Form.Label>
            <Button 
              variant="outline-secondary" 
              className="custom-button w-100 mb-2"
              onClick={() => document.getElementById('datasetInput').click()}
            >
              Upload dataset
            </Button>
            <Form.Control
              id="datasetInput"
              type="file"
              onChange={(e) => {
                if (e.target.files[0]) {
                  setValidationDataset(e.target.files[0]);
                }
              }}
              className="d-none"
            />
            {validationDataset && (
              <div className="d-flex align-items-center mb-2 p-2 bg-light rounded">
                <span className="text-truncate flex-grow-1">{validationDataset.name}</span>
                <button 
                  className="btn btn-sm text-danger" 
                  onClick={() => setValidationDataset(null)}
                  style={{ padding: '0 6px' }}
                  title="Remove file"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                    <path d="M4.646 4.646a.5.5 0 0 1 .708 0L8 7.293l2.646-2.647a.5.5 0 0 1 .708.708L8.707 8l2.647 2.646a.5.5 0 0 1-.708.708L8 8.707l-2.646 2.647a.5.5 0 0 1-.708-.708L7.293 8 4.646 5.354a.5.5 0 0 1 0-.708z"/>
                  </svg>
                </button>
              </div>
            )}
            <Form.Text className="text-muted">
              The validation dataset is used to measure the model's performance before and after optimization.
            </Form.Text>
          </Form.Group>
          
          {/* Submit Button - Centered */}
          <div className="d-flex justify-content-center mt-4">
            <Button 
              variant="secondary" 
              onClick={handleSubmit}
              className="custom-button px-4"
              disabled={!isFormValid}
              style={{ opacity: isFormValid ? 1 : 0.3, }}
            >
              Show Optimizations
            </Button>
          </div>
        </Form>
      </div>
    </Container>
  );
};

export default InputPage;
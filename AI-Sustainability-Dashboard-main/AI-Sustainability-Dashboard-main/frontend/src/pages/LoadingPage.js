import React from 'react';
import { Row, Col, Button } from 'react-bootstrap';
import LoadingSpinner from '../components/ui/LoadingSpinner';

/**
 * Page component for the loading screen
 * Shows loading spinner and current processing message
 * Content is centered both horizontally and vertically
 * With consistent background color
 */
const LoadingPage = ({ message, handleCancel }) => {
  return (
    <Row className="justify-content-center text-center align-items-center min-vh-100" style={{ backgroundColor: '#FDFDF7' }}>
      <Col md={6} className="my-auto">
        <LoadingSpinner message={message} />
        
        <Button
          variant="dark"
          onClick={handleCancel}
          className="rounded-pill px-5 py-2 mt-4"
        >
          Cancel
        </Button>
      </Col>
    </Row>
  );
};

export default LoadingPage;
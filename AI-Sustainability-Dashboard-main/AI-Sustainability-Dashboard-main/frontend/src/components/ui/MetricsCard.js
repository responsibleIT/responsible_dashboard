import React from 'react';
import { Card } from 'react-bootstrap';

/**
 * Card component to display performance metrics
 * Accepts type (for color styling) and value
 */
const MetricsCard = ({ type, title, value }) => {
  const colorMap = {
    power: 'primary',
    accuracy: 'success',
    carbon: 'warning',
    computing: 'danger'
  };
  
  const textClass = `text-${colorMap[type] || 'primary'}`;
  
  return (
    <Card className="h-100 bg-light border-0">
      <Card.Body>
        <Card.Title className={`${textClass} small`}>{title}</Card.Title>
        <h3 className={`${textClass} fw-bold`}>{value}</h3>
      </Card.Body>
    </Card>
  );
};

export default MetricsCard;
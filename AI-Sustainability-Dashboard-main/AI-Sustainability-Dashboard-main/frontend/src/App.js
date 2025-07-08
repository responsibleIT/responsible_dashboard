import React, { useState } from 'react';
import InputPage from './pages/InputPage';
import LoadingPage from './pages/LoadingPage';
import ResultsPage from './pages/ResultsPage';
import AppContainer from './components/layout/AppContainer';
import BenchmarkResultsPage from './pages/BenchmarkResultsPage';
import { fetchModelOptimizations, benchmarkModel } from './services/api';
import { benchmarkData } from './data/benchmark';

function App() {
  const [currentView, setCurrentView] = useState('url-input');
  const [modelUrl, setModelUrl] = useState('');
  const [loadingMessage, setLoadingMessage] = useState('');
  const [benchmarkMessage, setBenchmarkMessage] = useState('Running benchmark...');
  const [modelData, setModelData] = useState(null);
  const [threshold, setThreshold] = useState(0.0);
  
  const handleInputSubmit = async (e) => {
    e.preventDefault();
    setCurrentView('loading');
    
    try {
      // Using our API service to handle the fetching and loading states
      const data = await fetchModelOptimizations(modelUrl, setLoadingMessage);
      setModelData(data);
      setCurrentView('results');
    } catch (error) {
      console.error('Error:', error);
      // Handle error case - could set an error state here
      setCurrentView('url-input');
    }
  };

  const handleValidate = async (threshold) => {
    setThreshold(threshold);
    setCurrentView('benchmark');
    
    try {
      // Using our API service to handle the fetching and loading states
      const data = await benchmarkModel(modelUrl, threshold, setBenchmarkMessage);
      setModelData(data);
      setCurrentView('benchmark-details');
    } catch (error) {
      console.error('Error:', error);
      // Handle error case - could set an error state here
      setCurrentView('results');
    }
  }
  
  const handleLoadingCancel = () => {
    setCurrentView('url-input');
    setLoadingMessage('');
  };

  const handleBenchmarkCancel = () => {
    setCurrentView('results');
    setBenchmarkMessage('');
  }
  
  const handleBack = () => {
    setCurrentView('url-input');
  };

  const handleBenchmarkBack = () => {
    setCurrentView('results');
  }
  
  // Render different pages based on the current view state
  const renderContent = () => {
    switch (currentView) {
      case 'url-input':
        return (
          <InputPage 
            modelUrl={modelUrl} 
            setModelUrl={setModelUrl} 
            handleSubmit={handleInputSubmit} 
          />
        );
      case 'loading':
        return (
          <LoadingPage 
            message={loadingMessage} 
            handleCancel={handleLoadingCancel} 
          />
        );
      case 'results':
        return (
          <ResultsPage 
            modelData={modelData} 
            handleBack={handleBack} 
            handleValidate={handleValidate}
          />
        );
      case 'benchmark':
        return (
          <LoadingPage 
            message={benchmarkMessage} 
            handleCancel={handleBenchmarkCancel} 
          />
        );
      case 'benchmark-details':

        const data = benchmarkData;

        const _baselineData = data.find(item => item.threshold === 0.0);
        const _validatedData = data.find(item => item.threshold === threshold);

        let hardwarePerformance = 17.4; // 17.4 TFLOPS
        let hardwarePower = 400; // 400W
        let efficiency = hardwarePerformance * 1e9 / hardwarePower; // 43.5 TFLOPS/W

        const validated_flops = _validatedData.flops;
        const validated_energy = validated_flops / efficiency; // Energy in Joules
        const validated_power = validated_energy / 3600000

        const baseline_flops = _baselineData.flops;
        const baseline_energy = baseline_flops / efficiency; // Energy in Joules
        const baseline_power = baseline_energy / 3600000

        const carbonEmissionPerKWh = 50; // gCO2/kWh

        const validated_carbon = validated_power * carbonEmissionPerKWh;
        const baseline_carbon = baseline_power * carbonEmissionPerKWh;

        const validatedData = {
          ..._validatedData,
          power: validated_power,
          carbonEmissions: validated_carbon,
        }

        const baselineData = {
          ..._baselineData,
          power: baseline_power,
          carbonEmissions: baseline_carbon,
        }

        return (
          <BenchmarkResultsPage
            modelName={"twitter-xlm-roberta-base-sentiment"}
            validatedData={validatedData}
            baselineData={baselineData}
            originalParams={baselineData.non_zero_params}
            gpuModel={"NVIDIA A100"}
            location={"France"}
            onBack={handleBenchmarkBack} 
            onExport={() => console.log('Exporting...')}
          />
        );
      default:
        return <InputPage />;
    }
  };

  return <AppContainer>{renderContent()}</AppContainer>;
}

export default App;
document.addEventListener('DOMContentLoaded', function() {
    // Button event listeners
    const showChartBtn = document.getElementById('show-chart');
    if (showChartBtn) {
        showChartBtn.addEventListener('click', toggleChart);
    }
    
    const analyzeBtn = document.getElementById('analyze-data');
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', analyzeData);
    }
    
    const settingsBtn = document.getElementById('open-settings');
    if (settingsBtn) {
        settingsBtn.addEventListener('click', openSettings);
    }
    
    // Initialize Chart.js if canvas exists
    const chartCanvas = document.getElementById('sample-chart');
    if (chartCanvas) {
        initializeChart(chartCanvas);
    }
});

// Toggle chart visibility
function toggleChart() {
    const chartContainer = document.getElementById('chart-container');
    if (chartContainer.style.display === 'none' || chartContainer.style.display === '') {
        chartContainer.style.display = 'block';
        this.textContent = 'Hide Chart';
    } else {
        chartContainer.style.display = 'none';
        this.textContent = 'Show Chart';
    }
}

// Sample function for analyze button
function analyzeData() {
    alert('Data analysis would happen here in a real application.');
}

// Sample function for settings button
function openSettings() {
    alert('Settings dialog would open here in a real application.');
}

// Initialize a sample chart
function initializeChart(canvas) {
    const ctx = canvas.getContext('2d');
    
    // Create a sample chart using Chart.js
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['January', 'February', 'March', 'April', 'May', 'June'],
            datasets: [{
                label: 'Sample Data',
                data: [12, 19, 3, 5, 2, 3],
                backgroundColor: [
                    'rgba(255, 99, 132, 0.5)',
                    'rgba(54, 162, 235, 0.5)',
                    'rgba(255, 206, 86, 0.5)',
                    'rgba(75, 192, 192, 0.5)',
                    'rgba(153, 102, 255, 0.5)',
                    'rgba(255, 159, 64, 0.5)'
                ],
                borderColor: [
                    'rgba(255, 99, 132, 1)',
                    'rgba(54, 162, 235, 1)',
                    'rgba(255, 206, 86, 1)',
                    'rgba(75, 192, 192, 1)',
                    'rgba(153, 102, 255, 1)',
                    'rgba(255, 159, 64, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}
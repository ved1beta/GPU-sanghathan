# server.py
from flask import Flask, render_template, jsonify
import threading
import socketio
import time
import json
import os
from collections import defaultdict, deque
import numpy as np

# Store metrics history
metrics_history = {
    'accuracy': deque(maxlen=100),
    'loss': deque(maxlen=100),
    'throughput': deque(maxlen=100),
    'compute_time': defaultdict(lambda: deque(maxlen=100)),
    'comm_time': defaultdict(lambda: deque(maxlen=100)),
    'gpu_util': defaultdict(lambda: deque(maxlen=100)),
    'memory_usage': defaultdict(lambda: deque(maxlen=100)),
    'timestamps': deque(maxlen=100)
}

# Global training state
training_state = {
    'epoch': 0,
    'iteration': 0,
    'status': 'idle',
    'start_time': None,
    'processes': {},
    'pipeline_stages': {},
    'latest_metrics': {}
}

# Setup for metric collection
metrics_lock = threading.Lock()
process_data = {}

def start_monitoring_server(port=5000):
    """Launch web server for real-time monitoring"""
    app = Flask(__name__, 
                static_folder=os.path.join(os.path.dirname(__file__), 'static'),
                template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
    sio = socketio.Server(cors_allowed_origins='*')
    app_wsgi = socketio.WSGIApp(sio, app)
    
    # Create directory for templates if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'templates'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), 'static'), exist_ok=True)
    
    # Create a simple HTML template
    with open(os.path.join(os.path.dirname(__file__), 'templates', 'index.html'), 'w') as f:
        f.write("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Distributed Training Monitor</title>
            <script src="https://cdn.socket.io/4.4.1/socket.io.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .container { display: flex; flex-wrap: wrap; }
                .chart-container { width: 48%; margin: 10px; }
                .big-chart { width: 98%; }
                .metrics { display: flex; flex-wrap: wrap; }
                .metric-card { width: 200px; margin: 10px; padding: 15px; 
                              border-radius: 5px; background-color: #f5f5f5; }
                .pipeline-view { margin: 20px 0; }
                h2 { color: #333; }
                .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
                .running { background-color: #d4edda; }
                .idle { background-color: #f8f9fa; }
                .error { background-color: #f8d7da; }
            </style>
        </head>
        <body>
            <h1>Distributed Training Monitor</h1>
            
            <div id="status" class="status idle">Status: Idle</div>
            
            <div class="metrics">
                <div class="metric-card">
                    <h3>Epoch</h3>
                    <div id="epoch">0</div>
                </div>
                <div class="metric-card">
                    <h3>Iteration</h3>
                    <div id="iteration">0</div>
                </div>
                <div class="metric-card">
                    <h3>Accuracy</h3>
                    <div id="accuracy">0%</div>
                </div>
                <div class="metric-card">
                    <h3>Loss</h3>
                    <div id="loss">0</div>
                </div>
                <div class="metric-card">
                    <h3>Throughput</h3>
                    <div id="throughput">0 samples/sec</div>
                </div>
                <div class="metric-card">
                    <h3>Training Time</h3>
                    <div id="time">00:00:00</div>
                </div>
            </div>
            
            <div class="container">
                <div class="chart-container">
                    <canvas id="accuracyChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="lossChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="throughputChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="timeChart"></canvas>
                </div>
            </div>
            
            <h2>Pipeline Visualization</h2>
            <div class="pipeline-view big-chart">
                <svg id="pipelineView" width="100%" height="200"></svg>
            </div>
            
            <script>
                const socket = io();
                let accuracyChart, lossChart, throughputChart, timeChart;
                
                // Initialize charts
                function initCharts() {
                    accuracyChart = new Chart(document.getElementById('accuracyChart'), {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Accuracy',
                                data: [],
                                borderColor: 'rgba(75, 192, 192, 1)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true,
                            scales: {
                                y: {
                                    beginAtZero: true,
                                    max: 100
                                }
                            }
                        }
                    });
                    
                    lossChart = new Chart(document.getElementById('lossChart'), {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Loss',
                                data: [],
                                borderColor: 'rgba(255, 99, 132, 1)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true
                        }
                    });
                    
                    throughputChart = new Chart(document.getElementById('throughputChart'), {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Throughput (samples/sec)',
                                data: [],
                                borderColor: 'rgba(54, 162, 235, 1)',
                                tension: 0.1
                            }]
                        },
                        options: {
                            responsive: true
                        }
                    });
                    
                    timeChart = new Chart(document.getElementById('timeChart'), {
                        type: 'bar',
                        data: {
                            labels: ['Compute', 'Communication'],
                            datasets: [{
                                label: 'Time (ms)',
                                data: [0, 0],
                                backgroundColor: [
                                    'rgba(75, 192, 192, 0.2)',
                                    'rgba(255, 99, 132, 0.2)'
                                ],
                                borderColor: [
                                    'rgba(75, 192, 192, 1)',
                                    'rgba(255, 99, 132, 1)'
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
                
                // Update pipeline visualization
                function updatePipelineView(data) {
                    const svg = d3.select("#pipelineView");
                    svg.selectAll("*").remove();
                    
                    const stages = Object.keys(data.pipeline_stages).length;
                    if (stages === 0) return;
                    
                    const width = svg.node().getBoundingClientRect().width;
                    const height = svg.node().getBoundingClientRect().height;
                    const stageWidth = width / stages;
                    
                    // Draw stage boxes
                    Object.entries(data.pipeline_stages).forEach(([stageId, stageData], i) => {
                        const x = i * stageWidth;
                        const g = svg.append("g")
                            .attr("transform", `translate(${x}, 0)`);
                        
                        // Stage box
                        g.append("rect")
                            .attr("width", stageWidth - 10)
                            .attr("height", 80)
                            .attr("rx", 5)
                            .attr("fill", `rgba(54, 162, 235, ${0.3 + 0.7 * (stageData.utilization || 0)})`);
                        
                        // Stage label
                        g.append("text")
                            .attr("x", (stageWidth - 10) / 2)
                            .attr("y", 25)
                            .attr("text-anchor", "middle")
                            .text(`Stage ${stageId}`);
                        
                        // Utilization label
                        g.append("text")
                            .attr("x", (stageWidth - 10) / 2)
                            .attr("y", 45)
                            .attr("text-anchor", "middle")
                            .text(`${Math.round((stageData.utilization || 0) * 100)}% util`);
                        
                        // Layers label
                        g.append("text")
                            .attr("x", (stageWidth - 10) / 2)
                            .attr("y", 65)
                            .attr("text-anchor", "middle")
                            .attr("font-size", "10px")
                            .text(stageData.layers || "");
                    });
                    
                    // Draw arrows between stages
                    for (let i = 0; i < stages - 1; i++) {
                        const x1 = (i + 1) * stageWidth - 5;
                        const x2 = (i + 1) * stageWidth + 5;
                        const y = 40;
                        
                        svg.append("line")
                            .attr("x1", x1)
                            .attr("y1", y)
                            .attr("x2", x2)
                            .attr("y2", y)
                            .attr("stroke", "black")
                            .attr("marker-end", "url(#arrow)");
                    }
                    
                    // Add arrow marker
                    svg.append("defs").append("marker")
                        .attr("id", "arrow")
                        .attr("viewBox", "0 -5 10 10")
                        .attr("refX", 8)
                        .attr("refY", 0)
                        .attr("markerWidth", 6)
                        .attr("markerHeight", 6)
                        .attr("orient", "auto")
                        .append("path")
                        .attr("d", "M0,-5L10,0L0,5")
                        .attr("fill", "black");
                    
                    // Data flow visualization below the pipeline
                    const dataFlow = svg.append("g")
                        .attr("transform", `translate(0, 100)`);
                    
                    dataFlow.append("rect")
                        .attr("width", width)
                        .attr("height", 30)
                        .attr("fill", "none")
                        .attr("stroke", "#ddd");
                    
                    // Animate data packets
                    for (let i = 0; i < 3; i++) {
                        dataFlow.append("rect")
                            .attr("width", 20)
                            .attr("height", 20)
                            .attr("y", 5)
                            .attr("fill", "rgba(255, 99, 132, 0.7)")
                            .attr("rx", 3)
                            .attr("x", i * 100 - 20)
                            .transition()
                            .duration(5000)
                            .ease(d3.easeLinear)
                            .attr("x", width + 20)
                            .on("end", function repeat() {
                                d3.select(this)
                                    .attr("x", -20)
                                    .transition()
                                    .duration(5000)
                                    .ease(d3.easeLinear)
                                    .attr("x", width + 20)
                                    .on("end", repeat);
                            });
                    }
                }
                
                // Format time as HH:MM:SS
                function formatTime(seconds) {
                    const hrs = Math.floor(seconds / 3600);
                    const mins = Math.floor((seconds % 3600) / 60);
                    const secs = Math.floor(seconds % 60);
                    return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
                }
                
                // Update metrics display
                function updateMetrics(data) {
                    document.getElementById('epoch').textContent = data.epoch;
                    document.getElementById('iteration').textContent = data.iteration;
                    document.getElementById('accuracy').textContent = `${(data.latest_metrics.accuracy || 0).toFixed(2)}%`;
                    document.getElementById('loss').textContent = (data.latest_metrics.loss || 0).toFixed(4);
                    document.getElementById('throughput').textContent = `${(data.latest_metrics.throughput || 0).toFixed(1)} samples/sec`;
                    
                    if (data.start_time) {
                        const elapsed = (Date.now() - data.start_time) / 1000;
                        document.getElementById('time').textContent = formatTime(elapsed);
                    }
                    
                    // Update status box
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = `Status: ${data.status}`;
                    statusEl.className = `status ${data.status.toLowerCase()}`;
                    
                    // Update charts
                    if (data.metrics_history) {
                        // Update accuracy chart
                        accuracyChart.data.labels = Array.from({length: data.metrics_history.accuracy.length}, (_, i) => i);
                        accuracyChart.data.datasets[0].data = data.metrics_history.accuracy;
                        accuracyChart.update();
                        
                        // Update loss chart
                        lossChart.data.labels = Array.from({length: data.metrics_history.loss.length}, (_, i) => i);
                        lossChart.data.datasets[0].data = data.metrics_history.loss;
                        lossChart.update();
                        
                        // Update throughput chart
                        throughputChart.data.labels = Array.from({length: data.metrics_history.throughput.length}, (_, i) => i);
                        throughputChart.data.datasets[0].data = data.metrics_history.throughput;
                        throughputChart.update();
                        
                        // Update time chart with averages
                        const avgCompute = data.metrics_history.compute_time ? 
                            Object.values(data.metrics_history.compute_time).reduce((sum, arr) => sum + (arr.length ? arr[arr.length-1] : 0), 0) / 
                            Object.keys(data.metrics_history.compute_time).length : 0;
                            
                        const avgComm = data.metrics_history.comm_time ? 
                            Object.values(data.metrics_history.comm_time).reduce((sum, arr) => sum + (arr.length ? arr[arr.length-1] : 0), 0) / 
                            Object.keys(data.metrics_history.comm_time).length : 0;
                            
                        timeChart.data.datasets[0].data = [avgCompute, avgComm];
                        timeChart.update();
                    }
                    
                    // Update pipeline visualization
                    updatePipelineView(data);
                }
                
                // Socket event handlers
                socket.on('connect', () => {
                    console.log('Connected to server');
                    initCharts();
                });
                
                socket.on('metrics_update', (data) => {
                    console.log('Received metrics update:', data);
                    updateMetrics(data);
                });
                
                socket.on('disconnect', () => {
                    console.log('Disconnected from server');
                });
            </script>
        </body>
        </html>
        """)
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/metrics')
    def get_metrics():
        with metrics_lock:
            return jsonify({
                'training_state': training_state,
                'metrics_history': {k: list(v) for k, v in metrics_history.items() if not isinstance(v, defaultdict)},
                'compute_time': {k: list(v) for k, v in metrics_history['compute_time'].items()},
                'comm_time': {k: list(v) for k, v in metrics_history['comm_time'].items()},
            })
    
    @sio.on('connect')
    def connect(sid, environ):
        # Send initial state
        with metrics_lock:
            data = {
                **training_state,
                'metrics_history': {k: list(v) for k, v in metrics_history.items() if not isinstance(v, defaultdict)},
                'compute_time': {k: list(v) for k, v in metrics_history['compute_time'].items()},
                'comm_time': {k: list(v) for k, v in metrics_history['comm_time'].items()},
            }
            sio.emit('metrics_update', data, room=sid)
        
    # Broadcast metrics periodically
    def broadcast_metrics():
        while True:
            with metrics_lock:
                data = {
                    **training_state,
                    'metrics_history': {k: list(v) for k, v in metrics_history.items() if not isinstance(v, defaultdict)},
                    'compute_time': {k: list(v) for k, v in metrics_history['compute_time'].items()},
                    'comm_time': {k: list(v) for k, v in metrics_history['comm_time'].items()},
                }
                sio.emit('metrics_update', data)
            time.sleep(1)
            
    # Start broadcasting thread        
    threading.Thread(target=broadcast_metrics, daemon=True).start()
    
    # Start server
    print(f"Starting monitoring server on port {port}")
    import eventlet
    eventlet.wsgi.server(eventlet.listen(('0.0.0.0', port)), app_wsgi)

# Functions to update metrics from training processes
def update_training_state(epoch, iteration, status='running'):
    """Update global training state"""
    with metrics_lock:
        training_state['epoch'] = epoch
        training_state['iteration'] = iteration
        training_state['status'] = status
        if status == 'running' and training_state['start_time'] is None:
            training_state['start_time'] = time.time() * 1000

def register_process(rank, world_size, stage_id=None, dp_rank=None, layers=None):
    """Register a new process"""
    with metrics_lock:
        training_state['processes'][rank] = {
            'stage_id': stage_id,
            'dp_rank': dp_rank,
            'last_seen': time.time(),
            'world_size': world_size
        }
        if stage_id is not None:
            training_state['pipeline_stages'][stage_id] = {
                'utilization': 0.0,
                'layers': layers or ''
            }

def update_metrics(rank, metrics_dict):
    """Update metrics from a process"""
    with metrics_lock:
        process_data[rank] = {**metrics_dict, 'timestamp': time.time()}
        
        # Update latest metrics
        if 'accuracy' in metrics_dict:
            metrics_history['accuracy'].append(metrics_dict['accuracy'])
            training_state['latest_metrics']['accuracy'] = metrics_dict['accuracy']
        
        if 'loss' in metrics_dict:
            metrics_history['loss'].append(metrics_dict['loss'])
            training_state['latest_metrics']['loss'] = metrics_dict['loss']
        
        if 'throughput' in metrics_dict:
            metrics_history['throughput'].append(metrics_dict['throughput'])
            training_state['latest_metrics']['throughput'] = metrics_dict['throughput']
        
        if 'compute_time' in metrics_dict:
            metrics_history['compute_time'][rank].append(metrics_dict['compute_time'])
        
        if 'comm_time' in metrics_dict:
            metrics_history['comm_time'][rank].append(metrics_dict['comm_time'])
        
        if 'gpu_util' in metrics_dict:
            metrics_history['gpu_util'][rank].append(metrics_dict['gpu_util'])
        
        if 'memory_usage' in metrics_dict:
            metrics_history['memory_usage'][rank].append(metrics_dict['memory_usage'])
        
        metrics_history['timestamps'].append(time.time())
        
        # Update pipeline stage utilization
        if rank in training_state['processes'] and training_state['processes'][rank]['stage_id'] is not None:
            stage_id = training_state['processes'][rank]['stage_id']
            if 'utilization' in metrics_dict:
                training_state['pipeline_stages'][stage_id]['utilization'] = metrics_dict['utilization']

def mark_training_complete(success=True):
    """Mark training as complete"""
    with metrics_lock:
        training_state['status'] = 'completed' if success else 'error'

if __name__ == "__main__":
    # For standalone testing
    start_monitoring_server(port=5000)
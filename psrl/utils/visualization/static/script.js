document.addEventListener('DOMContentLoaded', function () {
    // PSRL Log Visualizer - Main JS
    // DOM Elements
    const dropArea = document.getElementById('drop-area');
    const inputFile = document.getElementById('input-file');
    const uploadForm = document.getElementById('upload-form');
    const uploadBtn = document.getElementById('upload-btn');
    const fileList = document.getElementById('file-list');
    const timelineContainer = document.getElementById('timeline-container');
    const emptyState = document.getElementById('empty-state');
    const loadingIndicator = document.getElementById('loading-indicator');
    const fileSelectorContainer = document.getElementById('file-selector-container');
    const fileSelector = document.getElementById('file-selector');
    const updateTimelineBtn = document.getElementById('update-timeline-btn');
    const statsContainer = document.getElementById('stats-container');
    const zoomInBtn = document.getElementById('zoom-in-btn');
    const zoomOutBtn = document.getElementById('zoom-out-btn');
    
    // Variables for state management
    let currentFiles = []; // Files selected for upload (not yet uploaded)
    let currentData = {}; // Timeline data returned from backend after analysis
    let xDomain = null; // X-axis domain for timeline
    let eventTypes = {}; // Event type definitions loaded from server
    let svg = null; // D3 SVG object for timeline
    let zoomFactor = 1; // Current zoom level for timeline
    let xScale = null; // D3 scale for x-axis
    let currentVisibleFiles = null; // Currently selected files for timeline display
    
    // Initialize drag-and-drop, event listeners, and event types
    initDragDrop();
    initEventListeners();
    loadEventTypes();
    
    // Set up drag-and-drop and file input for file selection
    function initDragDrop() {
        // Clicking the drop area opens the file selector dialog
        dropArea.addEventListener('click', () => {
            inputFile.click();
        });
        // When files are selected via the file dialog
        inputFile.addEventListener('change', handleFileSelection);
        // Prevent default browser behavior for drag-and-drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });
        // Highlight drop area when dragging files over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        // Remove highlight when dragging leaves or files are dropped
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        // Handle files dropped into the drop area
        dropArea.addEventListener('drop', handleDrop, false);
    }
    
    // Set up event listeners for form submission, timeline update, and zoom controls
    function initEventListeners() {
        // When the upload form is submitted
        uploadForm.addEventListener('submit', function (e) {
            e.preventDefault();
            if (currentFiles.length === 0) {
                showToast('Please select files to upload', 'error');
                return;
            }
            uploadAndAnalyze();
        });
        // When the user updates which files to show in the timeline
        updateTimelineBtn.addEventListener('click', () => {
            const visibleFiles = Array.from(fileSelector.querySelectorAll('input[type="checkbox"]:checked'))
                .map(checkbox => checkbox.value);
            if (visibleFiles.length === 0) {
                showToast('Please select at least one file to display', 'warning');
                return;
            }
            currentVisibleFiles = visibleFiles;
            renderTimeline(currentData, visibleFiles, zoomFactor);
            updateStatistics(window.lastStatsData, visibleFiles);
        });
        // Zoom in on the timeline
        zoomInBtn.addEventListener('click', () => {
            zoomFactor = Math.min(zoomFactor * 1.2, 10);
            const visibleFiles = getCurrentVisibleFiles();
            renderTimeline(currentData, visibleFiles, zoomFactor);
            updateStatistics(window.lastStatsData, visibleFiles);
        });
        // Zoom out on the timeline
        zoomOutBtn.addEventListener('click', () => {
            zoomFactor = Math.max(zoomFactor / 1.2, 0.5);
            const visibleFiles = getCurrentVisibleFiles();
            renderTimeline(currentData, visibleFiles, zoomFactor);
            updateStatistics(window.lastStatsData, visibleFiles);
        });
    }
    
    // Load event type definitions (for coloring and labeling events) from the server
    function loadEventTypes() {
        d3.json('/static/event_types.json').then(data => {
            eventTypes = data;
        }).catch(error => {
            console.error('Error loading event types:', error);
        });
    }
    
    // Prevent default drag-and-drop browser behavior
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    // Add highlight style to drop area
    function highlight() {
        dropArea.classList.add('border-primary', 'bg-blue-50');
    }
    
    // Remove highlight style from drop area
    function unhighlight() {
        dropArea.classList.remove('border-primary', 'bg-blue-50');
    }
    
    // Handle files dropped into the drop area
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }
    
    // Handle files selected via the file input dialog
    function handleFileSelection(e) {
        const files = e.target.files;
        handleFiles(files);
    }
    
    // Add new files to the upload list, avoiding duplicates by file name
    function handleFiles(files) {
        if (files.length === 0) return;
        // Append new files, avoid duplicates by name
        const existingNames = new Set(currentFiles.map(f => f.name));
        Array.from(files).forEach(file => {
            if (!existingNames.has(file.name)) {
                currentFiles.push(file);
                existingNames.add(file.name);
            }
        });
        renderFileList();
        uploadBtn.disabled = currentFiles.length === 0;
    }
    
    // Render the list of files selected for upload, with a remove button for each
    function renderFileList() {
        fileList.innerHTML = '';
        currentFiles.forEach((file, idx) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'flex items-center justify-between bg-gray-50 p-3 rounded-lg';
            fileItem.innerHTML = `
                <div class="flex items-center">
                    <i class="fa fa-file-text-o text-gray-400 mr-2"></i>
                    <span class="text-gray-700 truncate max-w-xs">${file.name}</span>
                </div>
                <div class="flex items-center space-x-2">
                    <span class="text-xs text-gray-500">${formatFileSize(file.size)}</span>
                    <button class="ml-2 text-red-400 hover:text-red-600 focus:outline-none" title="Remove" onclick="window._removeUploadFile(${idx})">
                        <i class="fa fa-times"></i>
                    </button>
                </div>
            `;
            fileList.appendChild(fileItem);
        });
    }
    
    // Remove a file from the upload list by index
    window._removeUploadFile = function(idx) {
        currentFiles.splice(idx, 1);
        renderFileList();
        uploadBtn.disabled = currentFiles.length === 0;
    };
    
    // Format file size in a human-readable way
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    // Upload selected files to the backend and trigger log analysis
    function uploadAndAnalyze() {
        // Hide empty state and show loading indicator
        emptyState.style.display = 'none';
        loadingIndicator.style.display = 'flex';
        statsContainer.style.display = 'none';
        fileSelectorContainer.style.display = 'none';
        
        const formData = new FormData();
        currentFiles.forEach(file => {
            formData.append('log_files', file);
        });
        
        // Send files to backend for analysis
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Server responded with error: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            if (!data.success) {
                throw new Error(data.error || 'Analysis failed');
            }
            // Store returned timeline and stats data
            currentData = data.data;
            // Create file checkboxes for timeline selection
            createFileCheckboxes(data.files);
            // Render the timeline visualization
            renderTimeline(currentData, null, zoomFactor);
            // Store and update statistics
            window.lastStatsData = data.stats_data;
            updateStatistics(data.stats_data, null);
            // Show file selector and stats
            fileSelectorContainer.style.display = 'block';
            statsContainer.style.display = 'block';
            showToast('Analysis completed successfully', 'success');
        })
        .catch(error => {
            console.error('Error analyzing logs:', error);
            showToast('Error analyzing logs: ' + error.message, 'error');
        })
        .finally(() => {
            loadingIndicator.style.display = 'none';
        });
    }
    
    // Create checkboxes for each file to allow user to select which files to show in the timeline
    function createFileCheckboxes(files) {
        fileSelector.innerHTML = '';
        files.forEach(file => {
            const checkbox = document.createElement('div');
            checkbox.className = 'flex items-center';
            checkbox.innerHTML = `
                <input type="checkbox" id="file-${file}" name="visible_files" value="${file}" checked 
                    class="w-4 h-4 text-primary focus:ring-primary border-gray-300 rounded">
                <label for="file-${file}" class="ml-2 text-sm font-medium text-gray-700">${file}</label>
            `;
            fileSelector.appendChild(checkbox);
        });
    }
    
    // Update the timeline based on currently selected files
    function updateTimeline() {
        const visibleFiles = Array.from(fileSelector.querySelectorAll('input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);
        if (visibleFiles.length === 0) {
            showToast('Please select at least one file to display', 'warning');
            return;
        }
        renderTimeline(currentData, visibleFiles, zoomFactor);
    }
    
    // Render the timeline visualization using D3.js
    function renderTimeline(timelineDataDict, visibleFiles = null, zoom = 1) {
        // Clear previous SVGs in label and timeline containers
        d3.select('#timeline-label-svg-container').selectAll('*').remove();
        d3.select('#timeline-svg-container').selectAll('svg').remove();
        d3.select('#timeline-svg-container .relative').remove();

        // If no files specified, show all files
        if (visibleFiles === null) {
            visibleFiles = Object.keys(timelineDataDict);
        }
        // Filter data to only include selected files
        const filteredData = {};
        visibleFiles.forEach(filename => {
            if (timelineDataDict[filename]) {
                filteredData[filename] = timelineDataDict[filename];
            }
        });
        // If no data, show empty state
        if (Object.keys(filteredData).length === 0) {
            document.getElementById('empty-state').style.display = 'flex';
            return;
        }
        document.getElementById('empty-state').style.display = 'none';
        // Always sort files by the timestamp of their first event (independent of zoom)
        const sortedFiles = Object.keys(filteredData).sort((a, b) => {
            const aFirst = filteredData[a][0];
            const bFirst = filteredData[b][0];
            const aTime = new Date(aFirst.start || aFirst.timestamp);
            const bTime = new Date(bFirst.start || bFirst.timestamp);
            return aTime - bTime;
        });
        // Gather all events for x-axis calculation
        const allEvents = [].concat(...Object.values(filteredData).map(data => data));
        if (allEvents.length === 0) {
            document.getElementById('empty-state').style.display = 'flex';
            return;
        }
        // Calculate x-axis (time) range
        let minTime = d3.min(allEvents, d => new Date(d.start || d.timestamp));
        let maxTime = d3.max(allEvents, d => new Date(d.end || d.timestamp));
        if (minTime.getTime() === maxTime.getTime()) {
            minTime = new Date(minTime.getTime() - 500);
            maxTime = new Date(maxTime.getTime() + 500);
        }
        xDomain = [minTime, maxTime];
        // Common parameters
        const labelWidth = 180;
        const margin = { top: 20, right: 20, bottom: 80, left: 20 };
        const rowHeight = 80;
        const height = Math.max(100, sortedFiles.length * rowHeight);
        // y-axis scale
        const y = d3.scaleBand()
            .domain(sortedFiles)
            .range([0, height])
            .padding(0.3);
        // 1. Draw label SVG (left)
        const labelSvg = d3.select('#timeline-label-svg-container')
            .append('svg')
            .attr('width', labelWidth)
            .attr('height', height + margin.top + margin.bottom);
        // Draw file name labels, vertically centered
        labelSvg.selectAll('.file-label')
            .data(sortedFiles)
            .enter()
            .append('text')
            .attr('class', 'file-label')
            .attr('x', labelWidth - 10)
            .attr('y', d => y(d) + y.bandwidth() / 2 + margin.top)
            .attr('dy', '0.35em')
            .attr('text-anchor', 'end')
            .attr('font-size', 14)
            .attr('fill', '#444')
            .text(d => d);
        // 2. Draw timeline SVG (right)
        const container = d3.select('#timeline-svg-container');
        const containerWidth = container.node().getBoundingClientRect().width;
        const baseWidth = Math.max(1000, containerWidth - margin.left - margin.right);
        const width = baseWidth * zoom;
        const svg = container.append('svg')
            .attr('width', width + margin.left + margin.right)
            .attr('height', height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        // x-axis scale
        xScale = d3.scaleTime()
            .domain(xDomain)
            .range([0, width]);
        // Prepare all events with their y position for rendering
        let allEventObjs = [];
        sortedFiles.forEach(filename => {
            const yPos = y(filename) + y.bandwidth() / 2;
            (filteredData[filename] || []).forEach(event => {
                allEventObjs.push({ ...event, yPos });
            });
        });
        // Draw event segments (rectangles for duration events)
        svg.selectAll('rect.timeline-segment')
            .data(allEventObjs.filter(e => e.event_class === 'segment'))
            .enter()
            .append('rect')
            .attr('class', 'timeline-segment cursor-pointer')
            .attr('x', d => xScale(new Date(d.start)))
            .attr('y', d => d.yPos - 0.3 * y.bandwidth())
            .attr('width', d => xScale(new Date(d.end)) - xScale(new Date(d.start)))
            .attr('height', 0.6 * y.bandwidth())
            .attr('fill', d => (eventTypes[d.event_type] || eventTypes.OTHER).color)
            .attr('opacity', 0.7)
            .append('title')
            .text(d => `${d.message}\nDuration: ${d.duration.toFixed(2)}s\nModel Version: ${d.model_version || 'N/A'}`);
        // Draw single events (points/circles)
        svg.selectAll('circle.timeline-point')
            .data(allEventObjs.filter(e => e.event_class === 'single'))
            .enter()
            .append('circle')
            .attr('class', 'timeline-point cursor-pointer')
            .attr('cx', d => xScale(new Date(d.timestamp)))
            .attr('cy', d => d.yPos)
            .attr('r', 6)
            .attr('fill', d => (eventTypes[d.event_type] || eventTypes.OTHER).color)
            .attr('opacity', 0.9)
            .append('title')
            .text(d => `${d.message}\nModel Version: ${d.model_version || 'N/A'}`);
        // Draw only the x-axis (time) at the bottom, no border or vertical grid lines
        const xAxis = d3.axisBottom(xScale)
            .tickFormat(d3.timeFormat('%H:%M:%S.%L'))
            .ticks(Math.min(10, width / 100));
        const xAxisGroup = svg.append('g')
            .attr('class', 'axis-x')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis);
        // Style x-axis text for readability
        const styleXAxisText = (selection) => {
            selection
                .attr('transform', 'rotate(45)')
                .attr('text-anchor', 'start')
                .attr('dx', '0.5em')
                .attr('dy', '0.35em')
                .attr('class', 'text-sm text-gray-600');
        };
        styleXAxisText(xAxisGroup.selectAll('text'));
        // Calculate and display total duration
        const totalSpanSeconds = (maxTime - minTime) / 1000;
        document.getElementById('total-duration').textContent = formatDuration(totalSpanSeconds);
    }
    
    // Update statistics (pie charts) for selected files
    function updateStatistics(statsData, visibleFiles = null) {
        if (!statsData) return;
        // Only show stats for selected files
        let filteredStats = statsData;
        if (visibleFiles && Array.isArray(visibleFiles)) {
            filteredStats = {};
            visibleFiles.forEach(f => {
                if (statsData[f]) filteredStats[f] = statsData[f];
            });
        }
        createPieCharts(filteredStats);
    }
    
    // Format a duration in seconds as a human-readable string
    function formatDuration(seconds) {
        if (seconds < 60) {
            return `${seconds.toFixed(2)}s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = seconds % 60;
            return `${minutes}m ${remainingSeconds.toFixed(0)}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const remainingMinutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${remainingMinutes}m`;
        }
    }
    
    // Create pie charts for event duration distribution per file
    function createPieCharts(statsData) {
        const container = d3.select('#stats-charts');
        container.selectAll('*').remove(); // Clear existing charts
        
        Object.entries(statsData).forEach(([filename, fileStats]) => {
            const chartContainer = container.append('div')
                .attr('class', 'bg-white p-4 rounded-lg shadow-sm')
                .style('min-height', '300px');
            
            // Add file title
            chartContainer.append('h4')
                .attr('class', 'text-md font-medium text-gray-700 mb-2')
                .text(`${filename} (Total: ${formatDuration(fileStats.total_duration_seconds)})`);
            
            // Create SVG for pie chart
            const width = 300;
            const height = 250;
            const radius = Math.min(width, height) / 2;
            
            const svg = chartContainer.append('svg')
                .attr('width', width)
                .attr('height', height)
                .append('g')
                .attr('transform', `translate(${width/2},${height/2})`);
            
            const data = fileStats.sorted_percentages || [];
            
            // Create color scale for event types
            const color = d3.scaleOrdinal()
                .domain(data.map(d => d[0]))
                .range(data.map(d => eventTypes[d[0]]?.color || eventTypes.OTHER.color));
            
            // Define pie layout
            const pie = d3.pie()
                .value(d => d[1])
                .sort(null);
            
            const arc = d3.arc()
                .innerRadius(0)
                .outerRadius(radius);
            
            // Create arcs for each event type
            const arcs = svg.selectAll('.arc')
                .data(pie(data))
                .enter()
                .append('g')
                .attr('class', 'arc');
            
            // Draw arcs
            arcs.append('path')
                .attr('d', arc)
                .attr('fill', d => color(d.data[0]))
                .on('mouseover', function(event, d) {
                    d3.select(this).attr('opacity', 0.7);
                    showPieTooltip(event, d, filename);
                })
                .on('mouseout', function() {
                    d3.select(this).attr('opacity', 1);
                    hidePieTooltip();
                });
            
            // Add percentage labels to large enough arcs
            arcs.filter(d => d.endAngle - d.startAngle > 0.2)
                .append('text')
                .attr('transform', d => `translate(${arc.centroid(d)})`)
                .attr('dy', '0.35em')
                .attr('text-anchor', 'middle')
                .attr('class', 'text-xs font-medium text-white drop-shadow-md')
                .text(d => `${d.data[1].toFixed(1)}%`);
        });
    }
    
    // Show tooltip for pie chart segment
    function showPieTooltip(event, data, filename) {
        const tooltip = d3.select('#stats-charts').append('div')
            .attr('class', 'tooltip absolute bg-white p-3 rounded-lg shadow-lg z-50 border border-gray-200 text-sm')
            .style('left', (event.pageX + 15) + 'px')
            .style('top', (event.pageY - 28) + 'px')
            .html(`
                <div class="font-medium text-gray-900">${filename}</div>
                <div class="text-gray-700">
                    <strong>${eventTypes[data.data[0]]?.label || data.data[0]}</strong><br>
                    ${data.data[1].toFixed(1)}%<br>
                </div>
            `);
    }
    
    // Hide pie chart tooltip
    function hidePieTooltip() {
        d3.selectAll('.tooltip').remove();
    }
    
    // Show a toast notification (info, success, warning, error)
    function showToast(message, type = 'info') {
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `fixed top-4 right-4 px-4 py-3 rounded-lg shadow-lg z-50 transform transition-all duration-500 opacity-0 translate-y-[-20px]`;
        // Toast header
        const header = `<div class="text-xs font-bold text-primary mb-1">PSRL Log Visualizer</div>`;
        // Set toast color based on type
        if (type === 'success') {
            toast.classList.add('bg-green-50', 'border-l-4', 'border-green-400', 'text-green-700');
            toast.innerHTML = header + `
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <i class="fa fa-check-circle text-green-500"></i>
                    </div>
                    <div class="ml-3">
                        <p class="text-sm font-medium">${message}</p>
                    </div>
                </div>
            `;
        } else if (type === 'error') {
            toast.classList.add('bg-red-50', 'border-l-4', 'border-red-400', 'text-red-700');
            toast.innerHTML = header + `
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <i class="fa fa-exclamation-circle text-red-500"></i>
                    </div>
                    <div class="ml-3">
                        <p class="text-sm font-medium">${message}</p>
                    </div>
                </div>
            `;
        } else if (type === 'warning') {
            toast.classList.add('bg-yellow-50', 'border-l-4', 'border-yellow-400', 'text-yellow-700');
            toast.innerHTML = header + `
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <i class="fa fa-exclamation-triangle text-yellow-500"></i>
                    </div>
                    <div class="ml-3">
                        <p class="text-sm font-medium">${message}</p>
                    </div>
                </div>
            `;
        } else {
            toast.classList.add('bg-blue-50', 'border-l-4', 'border-blue-400', 'text-blue-700');
            toast.innerHTML = header + `
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <i class="fa fa-info-circle text-blue-500"></i>
                    </div>
                    <div class="ml-3">
                        <p class="text-sm font-medium">${message}</p>
                    </div>
                </div>
            `;
        }
        // Add toast to body
        document.body.appendChild(toast);
        // Animate in
        setTimeout(() => {
            toast.classList.remove('opacity-0', 'translate-y-[-20px]');
        }, 10);
        // Animate out and remove after 3 seconds
        setTimeout(() => {
            toast.classList.add('opacity-0', 'translate-y-[-20px]');
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 500);
        }, 3000);
    }
    
    // Get the currently selected files for timeline display
    function getCurrentVisibleFiles() {
        if (currentVisibleFiles && currentVisibleFiles.length > 0) {
            return currentVisibleFiles;
        }
        // If not recorded, get from checkboxes
        const visibleFiles = Array.from(fileSelector.querySelectorAll('input[type="checkbox"]:checked'))
            .map(checkbox => checkbox.value);
        return visibleFiles.length > 0 ? visibleFiles : Object.keys(currentData);
    }
});

// 在文件末尾追加样式
const style = document.createElement('style');
style.innerHTML = `
#file-labels {
    min-width: 180px;
    position: sticky;
    left: 0;
    z-index: 10;
    background: white;
    border-right: 1px solid #eee;
}
.file-label-row {
    font-size: 14px;
    color: #444;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 10px;
}
`;
document.head.appendChild(style);    
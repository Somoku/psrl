# Log Timeline Visualization Tool

Log Timeline Visualization Tool is a powerful log analysis tool that helps you visually analyze system logs, identify performance bottlenecks, and track model version changes.

## Features

1. **Multi-file Support**: Analyze multiple log files simultaneously, compare events from different sources on separate timeline tracks, and identify system interactions and dependencies.

2. **Intuitive Visualization**: Use different colors and symbols to distinguish event types, with interactive tooltips displaying detailed information for quickly identifying performance bottlenecks.

3. **Event Matching Algorithm**: Uses a backward matching strategy to ensure each Begin event is correctly matched with the nearest End event of the same type.

4. **Intelligent Caching System**: Automatically caches analyzed log file content to avoid redundant processing and significantly improve analysis efficiency.

5. **Event Duration Statistics**: Analyzes the proportion of time spent on each event type and generates intuitive statistical charts to help identify the most time-consuming operations.

6. **Interactive Interface**: The web interface allows you to toggle visibility of timelines for specific files, making comparison and analysis easier.

7. **Data Export**: Supports exporting analysis results as high-quality images and structured JSON data for further analysis or sharing.

## Installation Dependencies

Before using this tool, you need to install the following dependencies:
```
pip install flask matplotlib pandas numpy
```

## Usage

### Command Line Mode

1. Analyze a log file directly:
   ```
   python log_visualizer.py /path/to/your/logfile.log
   ```

2. Analyze multiple log files:
   ```
   python log_visualizer.py /path/to/log1.log /path/to/log2.log
   ```

3. Analyze all log files in a directory:
   ```
   python log_visualizer.py /path/to/logs/
   ```

4. Specify output files:
   ```
   python log_visualizer.py /path/to/logs/ --output-image my_timeline.png --output-json timeline_data.json
   ```

### Web Mode

1. Start the web server:
   ```
   python log_visualizer.py --web --port 5000
   ```

2. Open in browser:
   ```
   http://localhost:5000
   ```

3. Upload log files via the web interface for analysis.

## Web Interface Features

- **File Upload**: Supports drag-and-drop or selecting multiple log files for analysis  
- **Timeline Control**: Use checkboxes to show/hide specific log file timelines  
- **Event Details**: Hover over events to view detailed info including timestamp, duration, and message  
- **Statistical Analysis**: Click the "View Event Statistics" button to view event duration stats  
- **Data Export**: Download the timeline as an image and export detailed data in JSON format  

## Log Format Requirements

This tool supports three types of event formats:

1. Begin Event:
   ```
   [Begin Event] EventType - Event message
   ```

2. End Event:
   ```
   [End Event] EventType - Event message - Time taken: 12.34 seconds
   ```

3. Single Point-in-Time Event:
   ```
   [Single Event] EventType - Event message
   ```

Supported event types (can be extended in the `EventType` class):
- PULL: Pull model  
- PUSH: Push model  
- BUFFER_READY: Buffer ready  
- INIT: Initialization  
- TRAIN: Training  
- GEN: Generation  
- VAL: Validation  
- WAIT: Waiting  
- OTHER: Other  

## Custom Extensions

You can extend the tool by:

1. Adding new event types and their display properties in the `EventType` class  
2. Modifying the regex patterns to support different log formats  
3. Adding new statistics features in `log_stats.py`  
4. Extending the web interface with more interactive capabilities  

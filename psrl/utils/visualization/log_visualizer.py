import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from flask import (
    Flask,
    jsonify,
    render_template,
    request,
)
from matplotlib.patches import Rectangle

from psrl.utils.visualization.event_type import EventType

# Regular expressions for different event types
BEGIN_EVENT_PATTERN = r"\[Begin Event\] (\w+) - (.*)"
END_EVENT_PATTERN = r"\[End Event\] (\w+) - (.*) - Time taken: ([\d.]+) seconds"
SINGLE_EVENT_PATTERN = r"\[Single Event\] (\w+) - (.*)"
MODEL_VERSION_PATTERN = r"model version (\d+)"

# Cache directory
CACHE_DIR = ".logviz_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_file_hash(file_path):
    """Calculate SHA-256 hash of a file"""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_from_cache(file_path):
    """Load parsed data from cache if available"""
    file_hash = get_file_hash(file_path)
    cache_file = os.path.join(CACHE_DIR, f"{file_hash}.json")

    if os.path.exists(cache_file):
        try:
            with open(cache_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache for {file_path}: {e}")
    return None


def save_to_cache(file_path, data):
    """Save parsed data to cache"""
    file_hash = get_file_hash(file_path)
    cache_file = os.path.join(CACHE_DIR, f"{file_hash}.json")

    try:
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving cache for {file_path}: {e}")
        return False


def parse_log_line(line):
    """Parse a single log line to extract timestamp, event type, and message"""
    # Parse timestamp
    timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)", line)
    if not timestamp_match:
        return None

    timestamp = datetime.strptime(timestamp_match.group(1), "%Y-%m-%d %H:%M:%S,%f")

    # Check for Begin Event
    begin_match = re.search(BEGIN_EVENT_PATTERN, line)
    if begin_match:
        event_type = begin_match.group(1)
        message = begin_match.group(2)
        model_version = extract_model_version(message)
        return {
            "timestamp": timestamp,
            "event_type": event_type,
            "message": message,
            "model_version": model_version,
            "event_class": "begin",
        }

    # Check for End Event
    end_match = re.search(END_EVENT_PATTERN, line)
    if end_match:
        event_type = end_match.group(1)
        message = end_match.group(2)
        duration = float(end_match.group(3))
        model_version = extract_model_version(message)
        return {
            "timestamp": timestamp,
            "event_type": event_type,
            "message": message,
            "model_version": model_version,
            "duration": duration,
            "event_class": "end",
        }

    # Check for Single Event
    single_match = re.search(SINGLE_EVENT_PATTERN, line)
    if single_match:
        event_type = single_match.group(1)
        message = single_match.group(2)
        model_version = extract_model_version(message)
        return {
            "timestamp": timestamp,
            "event_type": event_type,
            "message": message,
            "model_version": model_version,
            "event_class": "single",
        }

    return None


def extract_model_version(message):
    """Extract model version from message"""
    match = re.search(MODEL_VERSION_PATTERN, message)
    if match:
        return int(match.group(1))
    return None


def parse_log_file(log_file):
    """Parse entire log file and extract event information with caching"""
    # Check cache first
    cached_data = load_from_cache(log_file)
    if cached_data:
        print(f"Loaded {log_file} from cache")
        # Convert timestamps back to datetime objects
        for item in cached_data:
            if "timestamp" in item:
                item["timestamp"] = datetime.strptime(item["timestamp"], "%Y-%m-%d %H:%M:%S.%f")
            if "start" in item:
                item["start"] = datetime.strptime(item["start"], "%Y-%m-%d %H:%M:%S.%f")
            if "end" in item:
                item["end"] = datetime.strptime(item["end"], "%Y-%m-%d %H:%M:%S.%f")
        return cached_data

    events = []

    with open(log_file) as f:
        for line in f:
            parsed = parse_log_line(line)
            # print(parsed)
            if parsed:
                events.append(parsed)

    # Save to cache
    cache_data = []
    for item in events:
        cache_item = item.copy()
        if "timestamp" in cache_item:
            cache_item["timestamp"] = cache_item["timestamp"].strftime("%Y-%m-%d %H:%M:%S.%f")
        cache_data.append(cache_item)
    save_to_cache(log_file, cache_data)

    return events


def create_timeline_data(events):
    """Create timeline data by pairing Begin and End events (from latest to earliest)"""
    timeline_data = []
    # Dictionary to track pending begin events by type, stored as lists (latest first)
    pending_events = defaultdict(list)

    # Process events in chronological order
    for event in events:
        if event["event_class"] == "begin":
            # Add begin event to the front of the list for its type
            pending_events[event["event_type"]].insert(0, event)

        elif event["event_class"] == "end":
            # Try to find the latest matching begin event
            if event["event_type"] in pending_events and pending_events[event["event_type"]]:
                # Pop the latest begin event (first in the list)
                begin_event = pending_events[event["event_type"]].pop(0)

                timeline_data.append(
                    {
                        "start": begin_event["timestamp"],
                        "end": event["timestamp"],
                        "duration": event["duration"],
                        "event_type": begin_event["event_type"],
                        "message": begin_event["message"],
                        "model_version": begin_event["model_version"],
                        "event_class": "segment",
                    }
                )
            else:
                # No matching begin event found, treat as single event
                timeline_data.append(
                    {
                        "timestamp": event["timestamp"],
                        "event_type": event["event_type"],
                        "message": event["message"],
                        "model_version": event["model_version"],
                        "event_class": "single",
                    }
                )

        elif event["event_class"] == "single":
            # Add single event directly
            timeline_data.append(
                {
                    "timestamp": event["timestamp"],
                    "event_type": event["event_type"],
                    "message": event["message"],
                    "model_version": event["model_version"],
                    "event_class": "single",
                }
            )

    # Neglect unpaired begin events for now
    # assert len(pending_events) == 0, f"Unpaired begin events: {pending_events}"

    return timeline_data


def plot_timeline_multifile(timeline_data_dict, output_file="timeline.png", visible_files=None):
    """Plot timeline with multiple files in separate tracks, optionally showing only selected files"""
    # If no files specified, show all
    if visible_files is None:
        visible_files = list(timeline_data_dict.keys())

    # Filter data to include only visible files
    filtered_data = {filename: timeline_data_dict[filename] for filename in visible_files}

    if not filtered_data:
        print("No visible files to plot.")
        return None

    # Sort files by their first event's timestamp
    sorted_files = sorted(
        filtered_data.keys(),
        key=lambda x: min(item.get("start", item.get("timestamp")) for item in filtered_data[x]),
    )

    # Create y positions for each file
    file_positions = {filename: i for i, filename in enumerate(sorted_files)}

    # Find the earliest and latest timestamps for x-axis
    all_events = [event for file_data in filtered_data.values() for event in file_data]
    if not all_events:
        print("No events found to plot.")
        return None

    # Determine x-axis limits
    min_time = min(event.get("start", event.get("timestamp")) for event in all_events)
    max_time = max(event.get("end", event.get("timestamp")) for event in all_events)

    fig, ax = plt.subplots(figsize=(15, 5 + len(filtered_data) * 2))
    width = (max_time - min_time).total_seconds() / 10
    height = 5 + len(filtered_data) * 2
    fig.set_size_inches(width, height)

    # Plot timeline for each file
    for filename, timeline_data in filtered_data.items():
        y_pos = file_positions[filename]

        # Add file name label
        ax.text(
            min_time,
            y_pos,
            f"{filename}",
            ha="left",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

        # Plot events
        for event in timeline_data:
            if event["event_class"] in ["segment", "incomplete"]:
                start = mdates.date2num(event["start"])
                end = mdates.date2num(event["end"])

                # Get event type properties
                event_type = event["event_type"]
                try:
                    event_props = EventType[event_type].value
                except KeyError:
                    event_props = EventType.OTHER.value

                # Create rectangle for segment
                rect = Rectangle(
                    (start, y_pos - 0.2),
                    end - start,
                    0.4,
                    facecolor=event_props["color"],
                    alpha=0.7 if event["event_class"] == "segment" else 0.3,
                    edgecolor="black",
                    linewidth=0.5,
                    label=event["message"],
                )
                ax.add_patch(rect)

                # Add model version text if available
                """
                if event['model_version'] is not None:
                    ax.text(
                        start, y_pos - 0.45,
                        f"v{event['model_version']}",
                        ha='center', va='center',
                        fontsize=7, fontweight='bold',
                        color='white',
                        bbox=dict(facecolor='red', alpha=0.8, boxstyle='round,pad=0.1')
                    )
                """

            elif event["event_class"] == "single":
                timestamp = mdates.date2num(event["timestamp"])

                # Get event type properties
                event_type = event["event_type"]
                try:
                    event_props = EventType[event_type].value
                except KeyError:
                    event_props = EventType.OTHER.value

                # Plot single event marker
                ax.plot(
                    timestamp,
                    y_pos,
                    marker=event_props["plot_marker"],
                    markersize=10,
                    color=event_props["color"],
                    alpha=0.9,
                    label=event["message"],
                )

                # Add model version text if available
                if event["model_version"] is not None:
                    ax.text(
                        timestamp,
                        y_pos + 0.15,
                        f"v{event['model_version']}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color="white",
                        bbox=dict(facecolor="red", alpha=0.8, boxstyle="round,pad=0.1"),
                    )

    # Set x-axis limits
    ax.set_xlim(min_time, max_time)

    # Set y-axis limits and labels
    ax.set_ylim(-1, len(sorted_files))
    ax.set_yticks(list(file_positions.values()))
    ax.set_yticklabels([])  # Hide numeric y labels

    # Set x-axis as time format
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.xticks(rotation=45)

    # Add grid for better readability
    ax.grid(True, axis="x", linestyle="--", alpha=0.7)

    # Add legend for event types
    legend_elements = []

    # Add segment types
    for event_type in EventType:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color=event_type.value["color"],
                lw=4,
                alpha=0.7,
                label=f"{event_type.value['label']} (Segment)",
            )
        )

    # Add single event markers
    for event_type in EventType:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=event_type.value["plot_marker"],
                color="w",
                markerfacecolor=event_type.value["color"],
                markersize=10,
                label=f"{event_type.value['label']} (Point)",
            )
        )

    ax.legend(handles=legend_elements, loc="upper right", ncol=2, fontsize="small")

    # Set title and axis labels
    plt.title("Multi-file System Timeline Visualization")
    plt.xlabel("Time")

    # Adjust layout
    plt.tight_layout()

    # Save image
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return output_file


def export_to_json(timeline_data_dict, output_file="timeline_data.json"):
    """Export timeline data to JSON format"""
    # Convert datetime objects to strings for JSON serialization
    serializable_data = {}
    for filename, data in timeline_data_dict.items():
        serializable_data[filename] = []
        for item in data:
            serializable_item = item.copy()
            if "start" in item:
                serializable_item["start"] = item["start"].strftime("%Y-%m-%d %H:%M:%S.%f")
            if "end" in item:
                serializable_item["end"] = item["end"].strftime("%Y-%m-%d %H:%M:%S.%f")
            if "timestamp" in item:
                serializable_item["timestamp"] = item["timestamp"].strftime("%Y-%m-%d %H:%M:%S.%f")
            serializable_data[filename].append(serializable_item)

    with open(output_file, "w") as f:
        json.dump(serializable_data, f, indent=4)


def get_all_log_files(paths):
    """Get all log files from given paths (files or directories)"""
    log_files = []

    for path in paths:
        if os.path.isfile(path):
            log_files.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for file in files:
                    log_files.append(os.path.join(root, file))

    return log_files


def analyze_log_files(
    log_sources,
    output_image="timeline.png",
    output_json="timeline_data.json",
    save_local=True,
):
    """Analyze multiple log files or directories and generate timeline visualization"""
    # Get all log files
    log_files = get_all_log_files(log_sources)

    if not log_files:
        print("No log files found to analyze")
        return None

    # Parse each log file
    timeline_data_dict = {}
    for log_file in log_files:
        try:
            events = parse_log_file(log_file)
            timeline_data = create_timeline_data(events)
            filename = os.path.basename(log_file)
            if timeline_data:
                timeline_data_dict[filename] = timeline_data
            print(f"Successfully parsed {log_file}")
        except Exception as e:
            print(f"Error parsing {log_file}: {e}")

    if not timeline_data_dict:
        print("No valid log files to analyze")
        return None

    # Plot timeline
    if save_local:
        image_path = plot_timeline_multifile(timeline_data_dict, output_image)
        print(f"Timeline visualization saved to {image_path}")
    else:
        image_path = None

    # Export to JSON
    export_to_json(timeline_data_dict, output_json)
    print(f"Timeline data saved to {output_json}")

    return timeline_data_dict


# Flask web application
current_dir = os.path.dirname(os.path.abspath(__file__))
static_folder_path = os.path.join(current_dir, "static")
app = Flask(__name__, static_folder=static_folder_path, static_url_path="/static")
app.config["UPLOAD_FOLDER"] = ".uploads"
app.config["OUTPUT_FOLDER"] = ".output"
app.config["STATS_FOLDER"] = ".stats"

# Create directories if they don't exist
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
os.makedirs(app.config["STATS_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze uploaded log files"""
    if "log_files" not in request.files:
        return jsonify({"success": False, "error": "No files uploaded"})

    files = request.files.getlist("log_files")
    if not files or all(file.filename == "" for file in files):
        return jsonify({"success": False, "error": "No files selected"})

    # Save uploaded files
    log_files = []
    for file in files:
        if file.filename:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            log_files.append(file_path)

    # Analyze log files
    output_image = os.path.join(app.config["OUTPUT_FOLDER"], "timeline.png")
    output_json = os.path.join(app.config["OUTPUT_FOLDER"], "timeline_data.json")

    timeline_data = analyze_log_files(log_files, output_image, output_json, save_local=False)

    if not timeline_data:
        return jsonify({"success": False, "error": "Analysis failed"})

    # Generate event duration statistics
    from psrl.utils.visualization.log_stats import generate_event_stats

    stats_image = os.path.join(app.config["STATS_FOLDER"], "event_stats.png")
    stats_data = generate_event_stats(timeline_data, stats_image)

    return jsonify(
        {
            "success": True,
            "stats_data": stats_data,
            "data": timeline_data,
            "files": list(timeline_data.keys()),
        }
    )


def main():
    """Main function for command line or web mode"""
    parser = argparse.ArgumentParser(description="Log Timeline Visualization Tool")
    parser.add_argument("log_sources", nargs="*", help="Log files or directories to analyze")
    parser.add_argument("--output-image", "-o", default="timeline.png", help="Output image file")
    parser.add_argument("--output-json", "-j", default="timeline_data.json", help="Output JSON file")
    parser.add_argument("--web", action="store_true", help="Run in web mode")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")

    args = parser.parse_args()

    if args.web:
        print(f"Starting web server on port {args.port}...")
        print(f"Visit http://localhost:{args.port} in your browser")
        app.run(host="0.0.0.0", port=args.port, debug=True)
    else:
        if not args.log_sources:
            print("Error: No log files or directories specified")
            parser.print_help()
            return

        analyze_log_files(args.log_sources, args.output_image, args.output_json)


if __name__ == "__main__":
    main()

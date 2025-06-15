import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from psrl.utils.visualization.event_type import EventType

def generate_event_stats(timeline_data_dict, output_file='event_stats.png'):
    """Generate statistics and subfigure pie charts of event durations per file."""
    stats_by_file = {}

    # Determine layout for subplots
    n_files = len(timeline_data_dict)
    n_cols = min(3, n_files)
    n_rows = int(np.ceil(n_files / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_files > 1 else [axes]

    for ax, (filename, data) in zip(axes, timeline_data_dict.items()):
        # Collect duration data for this file
        event_durations = defaultdict(list)
        for item in data:
            if item.get('event_class') == 'segment':
                etype = item['event_type']
                event_durations[etype].append(item['duration'])

        # Calculate totals and percentages
        total_durations = {etype: sum(durs) for etype, durs in event_durations.items()}
        overall_total = sum(total_durations.values())
        percentages = {etype: (dur / overall_total) * 100 for etype, dur in total_durations.items()}
        sorted_perc = sorted(percentages.items(), key=lambda x: x[1], reverse=True)

        # Prepare labels, sizes, colors
        labels = [EventType[etype].value['label'] if etype in EventType.__members__ else etype
                  for etype, _ in sorted_perc]
        sizes = [perc for _, perc in sorted_perc]
        colors = [EventType[etype].value['color'] if etype in EventType.__members__ else '#6B7280'
                  for etype, _ in sorted_perc]

        # Plot pie
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90
        )
        ax.axis('equal')
        ax.set_title(f"{filename}\nTotal: {overall_total:.2f}s", fontsize=12)
        plt.setp(autotexts, size=9, weight="bold")
        plt.setp(texts, size=10)

        # Store stats
        stats_by_file[filename] = {
            'total_duration_seconds': overall_total,
            'duration_by_event_type': total_durations,
            'percentage_by_event_type': percentages,
            'sorted_percentages': sorted_perc,
        }

    # Hide unused axes
    for ax in axes[len(timeline_data_dict):]:
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return stats_by_file

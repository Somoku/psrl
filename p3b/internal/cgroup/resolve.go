// Resolution of the cgroup hierarchies this process can reach.
//
// The two files that answer it are /proc/self/mounts (which hierarchy is mounted
// where) and /proc/self/cgroup (which cgroup this process is in, per hierarchy).
// Neither alone is enough: the mount tells us the controllers, and the process file
// tells us the path, and under a cgroup namespace the two disagree because the path
// is stated relative to the host while the mount shows only this container's subtree.
package cgroup

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// mount is one cgroup mount from /proc/self/mounts.
type mount struct {
	filesystem  string // "cgroup" or "cgroup2"
	point       string // where it is mounted
	controllers []string
}

// readMounts parses the cgroup and cgroup2 mounts, in file order.
func readMounts(path string) ([]mount, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cgroup: reading %s: %w", path, err)
	}
	var out []mount
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 4 {
			continue
		}
		filesystem := fields[2]
		if filesystem != "cgroup" && filesystem != "cgroup2" {
			continue
		}
		entry := mount{filesystem: filesystem, point: unescapeMountPoint(fields[1])}
		for _, option := range strings.Split(fields[3], ",") {
			if _, known := knownControllers[option]; known {
				entry.controllers = append(entry.controllers, option)
			}
		}
		if filesystem == "cgroup2" {
			entry.controllers = readUnifiedControllers(entry.point)
		}
		out = append(out, entry)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("cgroup: %s has no cgroup or cgroup2 mount", path)
	}
	return out, nil
}

// readUnifiedControllers reads the controllers a v2 root can hand to children.
//
// The mount's option list does not name them on v2; cgroup.controllers does, and it is
// the only place that says which ones are actually available rather than compiled in.
func readUnifiedControllers(point string) []string {
	raw, err := os.ReadFile(filepath.Join(point, "cgroup.controllers"))
	if err != nil {
		return nil
	}
	return strings.Fields(string(raw))
}

// readProcessPaths maps a controller list to this process's cgroup path, from
// /proc/self/cgroup. Both the joint spelling ("cpu,cpuacct") and each member are
// recorded, because a v1 mount can be named either way.
func readProcessPaths(path string) (map[string]string, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cgroup: reading %s: %w", path, err)
	}
	out := map[string]string{}
	for _, line := range strings.Split(string(raw), "\n") {
		parts := strings.SplitN(strings.TrimSpace(line), ":", 3)
		if len(parts) != 3 {
			continue
		}
		out[parts[1]] = parts[2]
		for _, controller := range strings.Split(parts[1], ",") {
			if controller != "" {
				out[controller] = parts[2]
			}
		}
		// A v2 entry states an empty controller list ("0::/path").
		if parts[1] == "" {
			out[""] = parts[2]
		}
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("cgroup: %s has no usable entries", path)
	}
	return out, nil
}

// resolveHierarchy works out this process's cgroup for one mount.
//
// The process's own path is tried first. Under a cgroup namespace it does not exist
// under the mount — the mount root *is* the process's cgroup — and that is the only
// case in which falling back to the root is right. A configured subtree is resolved
// separately and never falls back, because a missing subtree is the error the design
// asks us to report rather than to paper over.
func resolveHierarchy(entry mount, paths map[string]string) (Hierarchy, bool) {
	path := ""
	if entry.filesystem == "cgroup2" {
		path = paths[""]
	} else {
		for _, controller := range entry.controllers {
			if found, ok := paths[controller]; ok {
				path = found
				break
			}
		}
	}
	root := entry.point
	if path != "" && path != "/" {
		candidate := filepath.Join(entry.point, path)
		if _, err := os.Stat(filepath.Join(candidate, "cgroup.procs")); err == nil {
			root = candidate
		}
	}
	if _, err := os.Stat(filepath.Join(root, "cgroup.procs")); err != nil {
		return Hierarchy{}, false
	}
	return Hierarchy{Controllers: entry.controllers, Mount: entry.point, Root: root}, true
}

// unescapeMountPoint undoes the octal escapes /proc/self/mounts applies to spaces,
// tabs and backslashes in a mount point.
func unescapeMountPoint(value string) string {
	if !strings.Contains(value, `\`) {
		return value
	}
	var out strings.Builder
	for i := 0; i < len(value); i++ {
		if value[i] == '\\' && i+3 < len(value) {
			digits := value[i+1 : i+4]
			if number, err := parseOctal(digits); err == nil {
				out.WriteByte(byte(number))
				i += 3
				continue
			}
		}
		out.WriteByte(value[i])
	}
	return out.String()
}

func parseOctal(digits string) (int, error) {
	value := 0
	for _, digit := range digits {
		if digit < '0' || digit > '7' {
			return 0, fmt.Errorf("not octal: %q", digits)
		}
		value = value*8 + int(digit-'0')
	}
	return value, nil
}

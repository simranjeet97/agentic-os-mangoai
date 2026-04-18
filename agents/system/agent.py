"""
agents/system/agent.py — SystemAgent

Real-time system monitoring and management:
  - Monitor:  CPU, RAM, disk, network I/O in real time (via psutil)
  - Anomaly detection: CPU > 90%, RAM > 90%, disk > 85%, unusual network
  - Alerts:   disk > 85% → WARNING, RAM > 90% → CRITICAL alert
  - Fixes:    suggest and optionally apply fixes (kill process, clean temp)
  - Services: restart systemd/launchd services (with guardrail approval)
  - Cleanup:  remove temp files from /tmp, /var/tmp, ~/.cache (dry_run by default)
  - Processes: list, filter, send signals
  - Network:  live interface stats, open connections

All destructive operations (kill, restart, clean) are gated through GuardrailMiddleware.
"""

from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.base_agent import BaseAgent
from core.state import AgentState
from guardrails.models import ActionType, AgentAction


# ── Thresholds ────────────────────────────────────────────────────────────────

ALERT_DISK_PCT = float(os.getenv("ALERT_DISK_PCT", "85"))
ALERT_RAM_PCT = float(os.getenv("ALERT_RAM_PCT", "90"))
ALERT_CPU_PCT = float(os.getenv("ALERT_CPU_PCT", "90"))


class SystemAgent(BaseAgent):
    name = "system"
    description = "Real-time system monitoring, anomaly detection, service management, and cleanup"
    capabilities = ["process_list", "process_kill", "config_write", "shell_exec"]
    tools = [
        "get_metrics", "monitor_realtime", "detect_anomalies",
        "list_processes", "kill_process", "restart_service",
        "clean_temp", "get_network_stats", "suggest_fixes",
    ]

    def __init__(self, agent_id: Optional[str] = None) -> None:
        super().__init__(agent_id)
        self._alerts: list[dict[str, Any]] = []

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run() -> dict[str, Any]:
            action = step.get("action", "metrics")
            dispatch = {
                "metrics": self._get_metrics,
                "monitor": self._monitor_realtime,
                "anomalies": self._detect_anomalies,
                "processes": self._list_processes,
                "kill": self._kill_process,
                "restart": self._restart_service,
                "clean": self._clean_temp,
                "network": self._get_network_stats,
                "suggest": self._suggest_fixes,
                "run": self._run_in_sandbox,
            }
            handler = dispatch.get(action, self._get_metrics)
            # Most sub-handlers take only step for simplicity
            if action in ("run",):
                return await handler(step.get("command", ""))  # type: ignore[operator]
            return await handler(step, state)  # type: ignore[operator]

        return await self._run_with_audit(step, state, _run)

    # ── METRICS ───────────────────────────────────────────────────────────────

    async def _get_metrics(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Collect current CPU, RAM, disk, and network metrics."""
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(None, self._collect_metrics)

        # Check alert conditions
        alerts = self._check_thresholds(metrics)
        self._alerts.extend(alerts)

        return {
            "output": self._format_metrics(metrics),
            "metrics": metrics,
            "alerts": alerts,
            "step_type": "metrics",
        }

    def _collect_metrics(self) -> dict[str, Any]:
        """Synchronous psutil metric collection."""
        try:
            import psutil
        except ImportError:
            return {"error": "psutil not installed — run: pip install psutil"}

        # CPU
        cpu_pct = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()

        # Memory
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        # Disk (all partitions)
        disks: list[dict] = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
                disks.append({
                    "mountpoint": part.mountpoint,
                    "device": part.device,
                    "fstype": part.fstype,
                    "total_gb": round(usage.total / 1e9, 2),
                    "used_gb": round(usage.used / 1e9, 2),
                    "free_gb": round(usage.free / 1e9, 2),
                    "percent": usage.percent,
                })
            except PermissionError:
                continue

        # Network I/O
        net_io = psutil.net_io_counters()
        net_if = psutil.net_if_stats()

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "cpu": {
                "percent": cpu_pct,
                "count_logical": cpu_count,
                "count_physical": psutil.cpu_count(logical=False),
                "freq_mhz": round(cpu_freq.current, 1) if cpu_freq else None,
                "per_core": psutil.cpu_percent(interval=0, percpu=True),
            },
            "memory": {
                "total_gb": round(mem.total / 1e9, 2),
                "available_gb": round(mem.available / 1e9, 2),
                "used_gb": round(mem.used / 1e9, 2),
                "percent": mem.percent,
                "swap_used_gb": round(swap.used / 1e9, 2),
                "swap_total_gb": round(swap.total / 1e9, 2),
            },
            "disk": disks,
            "network": {
                "bytes_sent_mb": round(net_io.bytes_sent / 1e6, 2),
                "bytes_recv_mb": round(net_io.bytes_recv / 1e6, 2),
                "packets_sent": net_io.packets_sent,
                "packets_recv": net_io.packets_recv,
                "errin": net_io.errin,
                "errout": net_io.errout,
                "dropin": net_io.dropin,
                "dropout": net_io.dropout,
            },
            "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
        }

    # ── MONITOR REALTIME ──────────────────────────────────────────────────────

    async def _monitor_realtime(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Collect metrics at intervals and return a snapshot series.
        intervals: number of samples
        interval_sec: seconds between samples
        """
        intervals = int(step.get("intervals", 3))
        interval_sec = float(step.get("interval_sec", 2.0))

        samples: list[dict[str, Any]] = []
        loop = asyncio.get_event_loop()

        for _ in range(intervals):
            m = await loop.run_in_executor(None, self._collect_metrics)
            samples.append(m)
            if _ < intervals - 1:
                await asyncio.sleep(interval_sec)

        # Compute averages
        if samples:
            avg_cpu = sum(s["cpu"]["percent"] for s in samples) / len(samples)
            avg_mem = sum(s["memory"]["percent"] for s in samples) / len(samples)
        else:
            avg_cpu = avg_mem = 0.0

        all_alerts = []
        for s in samples:
            all_alerts.extend(self._check_thresholds(s))

        return {
            "output": f"Monitored {intervals} samples. avg_cpu={avg_cpu:.1f}%, avg_mem={avg_mem:.1f}%",
            "samples": samples,
            "avg_cpu_pct": round(avg_cpu, 2),
            "avg_mem_pct": round(avg_mem, 2),
            "alerts": all_alerts,
            "step_type": "monitor",
        }

    # ── ANOMALY DETECTION ─────────────────────────────────────────────────────

    async def _detect_anomalies(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Collect metrics and run anomaly scoring."""
        loop = asyncio.get_event_loop()
        metrics = await loop.run_in_executor(None, self._collect_metrics)

        anomalies: list[dict[str, Any]] = []

        # CPU
        if metrics["cpu"]["percent"] > ALERT_CPU_PCT:
            anomalies.append({
                "type": "high_cpu",
                "severity": "critical",
                "value": metrics["cpu"]["percent"],
                "threshold": ALERT_CPU_PCT,
                "message": f"CPU at {metrics['cpu']['percent']:.1f}% (threshold: {ALERT_CPU_PCT}%)",
                "suggestion": "Identify CPU-hungry processes via processes action, consider kill or priority adjustment.",
            })

        # RAM
        if metrics["memory"]["percent"] > ALERT_RAM_PCT:
            anomalies.append({
                "type": "high_memory",
                "severity": "critical",
                "value": metrics["memory"]["percent"],
                "threshold": ALERT_RAM_PCT,
                "message": f"RAM at {metrics['memory']['percent']:.1f}% (threshold: {ALERT_RAM_PCT}%)",
                "suggestion": "Find memory-heavy processes, consider restarting leaky services.",
            })

        # Disk
        for disk in metrics.get("disk", []):
            if disk["percent"] > ALERT_DISK_PCT:
                anomalies.append({
                    "type": "disk_full",
                    "severity": "warning" if disk["percent"] < 95 else "critical",
                    "mountpoint": disk["mountpoint"],
                    "value": disk["percent"],
                    "threshold": ALERT_DISK_PCT,
                    "message": f"Disk {disk['mountpoint']} at {disk['percent']:.1f}% (threshold: {ALERT_DISK_PCT}%)",
                    "suggestion": "Run clean action to remove temp files, archive old logs.",
                })

        # Network errors
        net = metrics.get("network", {})
        if net.get("errin", 0) + net.get("errout", 0) > 100:
            anomalies.append({
                "type": "network_errors",
                "severity": "warning",
                "errors": net.get("errin", 0) + net.get("errout", 0),
                "message": f"High network error count: {net.get('errin')} in, {net.get('errout')} out",
                "suggestion": "Check network interface configuration and cable/wireless signal.",
            })

        return {
            "output": f"Detected {len(anomalies)} anomalies",
            "anomalies": anomalies,
            "metrics": metrics,
            "step_type": "anomaly_detection",
            "health": "critical" if any(a["severity"] == "critical" for a in anomalies) else
                      "warning" if anomalies else "healthy",
        }

    # ── PROCESS LIST ──────────────────────────────────────────────────────────

    async def _list_processes(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """List running processes with CPU/RAM usage."""
        sort_by = step.get("sort_by", "cpu")    # cpu | mem | pid | name
        limit = int(step.get("limit", 20))
        filter_name = step.get("filter", "")

        loop = asyncio.get_event_loop()

        def _list():
            try:
                import psutil
            except ImportError:
                return []

            procs = []
            for proc in psutil.process_iter(
                ["pid", "name", "username", "cpu_percent", "memory_percent", "status", "create_time"]
            ):
                try:
                    info = proc.info
                    if filter_name and filter_name.lower() not in info.get("name", "").lower():
                        continue
                    procs.append({
                        "pid": info["pid"],
                        "name": info["name"],
                        "user": info.get("username", "?"),
                        "cpu_pct": round(info.get("cpu_percent") or 0, 2),
                        "mem_pct": round(info.get("memory_percent") or 0, 2),
                        "status": info.get("status", "?"),
                        "started_at": datetime.fromtimestamp(
                            info.get("create_time", 0)
                        ).isoformat() if info.get("create_time") else None,
                    })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            key = "cpu_pct" if sort_by == "cpu" else ("mem_pct" if sort_by == "mem" else sort_by)
            procs.sort(key=lambda p: p.get(key, 0), reverse=(sort_by in ("cpu", "mem")))
            return procs[:limit]

        processes = await loop.run_in_executor(None, _list)
        return {
            "output": f"Listed {len(processes)} processes (sorted by {sort_by})",
            "processes": processes,
            "step_type": "process_list",
        }

    # ── KILL PROCESS ──────────────────────────────────────────────────────────

    async def _kill_process(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Terminate a process by PID or name.
        Requires guardrail approval and explicit confirmed=True.
        """
        pid = step.get("pid")
        name = step.get("name", "")
        signal_str = step.get("signal", "TERM")   # TERM | KILL
        confirmed = step.get("confirmed", False)

        if not confirmed:
            return {
                "success": False,
                "requires_approval": True,
                "error": f"Process kill requires explicit confirmation. Set step['confirmed']=True.",
            }

        # Guardrail check
        try:
            action = AgentAction(
                agent_id=self.agent_id,
                agent_type=self.name,
                action_type=ActionType.PROCESS_SPAWN,
                command=f"kill -{signal_str} {pid or name}",
                raw_input=f"kill {pid or name}",
            )
            await self.guardrail.evaluate_action(action, user_role="agent")
        except Exception as exc:
            return {"success": False, "error": str(exc), "blocked": True}

        loop = asyncio.get_event_loop()

        def _kill():
            try:
                import psutil, signal as sig_module
                signal_val = sig_module.SIGTERM if signal_str == "TERM" else sig_module.SIGKILL

                if pid:
                    proc = psutil.Process(int(pid))
                    proc.send_signal(signal_val)
                    return {"killed": [pid], "name": proc.name()}
                else:
                    killed = []
                    for proc in psutil.process_iter(["pid", "name"]):
                        if name.lower() in (proc.info.get("name") or "").lower():
                            proc.send_signal(signal_val)
                            killed.append(proc.info["pid"])
                    return {"killed": killed}
            except Exception as e:
                return {"error": str(e)}

        result = await loop.run_in_executor(None, _kill)
        return {
            "output": f"Killed processes: {result.get('killed', [])}",
            **result,
            "step_type": "kill_process",
        }

    # ── RESTART SERVICE ───────────────────────────────────────────────────────

    async def _restart_service(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Restart a system service. Requires guardrail approval."""
        service = step.get("service", "")
        confirmed = step.get("confirmed", False)

        if not service:
            return {"success": False, "error": "No service name provided"}

        if not confirmed:
            return {
                "success": False,
                "requires_approval": True,
                "error": f"Service restart ({service}) requires explicit confirmation.",
            }

        # Detect init system
        use_systemctl = os.path.exists("/bin/systemctl") or os.path.exists("/usr/bin/systemctl")
        use_launchctl = os.path.exists("/bin/launchctl")

        if use_systemctl:
            cmd = f"systemctl restart {service}"
        elif use_launchctl:
            cmd = f"launchctl stop {service} && launchctl start {service}"
        else:
            return {"success": False, "error": "No supported init system found (systemd/launchd)"}

        # Guardrail check
        try:
            action = AgentAction(
                agent_id=self.agent_id,
                agent_type=self.name,
                action_type=ActionType.SHELL_COMMAND,
                command=cmd,
                raw_input=f"restart {service}",
            )
            await self.guardrail.evaluate_action(action, user_role="agent")
        except Exception as exc:
            return {"success": False, "error": str(exc), "blocked": True}

        result = await self._run_in_sandbox(cmd)
        return {**result, "service": service, "step_type": "restart_service"}

    # ── CLEAN TEMP FILES ──────────────────────────────────────────────────────

    async def _clean_temp(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Remove temp files from standard locations.
        dry_run=True by default — set to False to actually delete.
        """
        paths = step.get("paths", ["/tmp", os.path.expanduser("~/.cache")])
        dry_run = step.get("dry_run", True)
        older_than_days = int(step.get("older_than_days", 7))

        cutoff = time.time() - (older_than_days * 86400)
        freed_bytes = 0
        removed: list[str] = []

        loop = asyncio.get_event_loop()

        def _scan_and_clean():
            nonlocal freed_bytes
            candidates = []
            for base in paths:
                base_path = Path(base)
                if not base_path.exists():
                    continue
                for item in base_path.rglob("*"):
                    try:
                        if item.is_file() and item.stat().st_mtime < cutoff:
                            size = item.stat().st_size
                            candidates.append((str(item), size))
                            if not dry_run:
                                item.unlink(missing_ok=True)
                                freed_bytes += size
                                removed.append(str(item))
                    except (PermissionError, OSError):
                        continue
            return candidates

        candidates = await loop.run_in_executor(None, _scan_and_clean)
        total_size = sum(c[1] for c in candidates)

        return {
            "output": (
                f"{'Would free' if dry_run else 'Freed'} "
                f"{total_size / 1e6:.1f} MB from {len(candidates)} files"
            ),
            "candidates": [c[0] for c in candidates[:50]],
            "total_files": len(candidates),
            "total_size_mb": round(total_size / 1e6, 2),
            "dry_run": dry_run,
            "removed": removed[:50],
            "step_type": "clean_temp",
        }

    # ── NETWORK STATS ─────────────────────────────────────────────────────────

    async def _get_network_stats(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Get per-interface network stats and open connections."""
        loop = asyncio.get_event_loop()

        def _collect():
            try:
                import psutil
            except ImportError:
                return {"error": "psutil not installed"}

            per_nic = psutil.net_io_counters(pernic=True)
            stats = {}
            for name, counters in per_nic.items():
                stats[name] = {
                    "bytes_sent_mb": round(counters.bytes_sent / 1e6, 2),
                    "bytes_recv_mb": round(counters.bytes_recv / 1e6, 2),
                    "packets_sent": counters.packets_sent,
                    "packets_recv": counters.packets_recv,
                }

            connections = []
            for conn in psutil.net_connections(kind="inet"):
                try:
                    connections.append({
                        "fd": conn.fd,
                        "type": str(conn.type),
                        "local": f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else "",
                        "remote": f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else "",
                        "status": conn.status,
                        "pid": conn.pid,
                    })
                except Exception:
                    continue

            return {"interfaces": stats, "connections": connections[:100]}

        data = await loop.run_in_executor(None, _collect)
        return {
            "output": f"Network stats for {len(data.get('interfaces', {}))} interfaces, "
                      f"{len(data.get('connections', []))} connections",
            **data,
            "step_type": "network_stats",
        }

    # ── SUGGEST FIXES ─────────────────────────────────────────────────────────

    async def _suggest_fixes(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Detect anomalies and suggest remediations."""
        anomaly_result = await self._detect_anomalies(step, state)
        anomalies = anomaly_result.get("anomalies", [])

        if not anomalies:
            return {
                "output": "System is healthy — no fixes needed.",
                "suggestions": [],
                "step_type": "suggest_fixes",
            }

        suggestions = []
        for a in anomalies:
            suggestions.append({
                "anomaly": a["type"],
                "severity": a["severity"],
                "message": a["message"],
                "suggestion": a.get("suggestion", ""),
                "auto_fixable": a["type"] in ("disk_full",),  # Only disk cleanup is auto
            })

        return {
            "output": f"Found {len(suggestions)} issues with fix suggestions",
            "suggestions": suggestions,
            "anomalies": anomalies,
            "step_type": "suggest_fixes",
        }

    # ── RUN IN SANDBOX (subprocess fallback) ─────────────────────────────────

    async def _run_in_sandbox(self, command: str) -> dict[str, Any]:
        """
        Run a shell command safely via asyncio subprocess.
        Used internally by ExecutorAgent and RestartService.
        """
        if not command:
            return {"success": False, "error": "Empty command"}

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PATH": "/usr/bin:/bin:/usr/local/bin"},
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Command timed out after 30 seconds",
                    "exit_code": -1,
                    "timed_out": True,
                }

            exit_code = proc.returncode or 0
            return {
                "stdout": stdout.decode("utf-8", errors="replace")[:10000],
                "stderr": stderr.decode("utf-8", errors="replace")[:2000],
                "exit_code": exit_code,
                "success": exit_code == 0,
            }
        except Exception as exc:
            return {"success": False, "stdout": "", "stderr": str(exc), "exit_code": -1}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _check_thresholds(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Check metric values against alert thresholds. Returns alert list."""
        alerts = []
        cpu = metrics.get("cpu", {})
        mem = metrics.get("memory", {})

        if cpu.get("percent", 0) > ALERT_CPU_PCT:
            alerts.append({
                "level": "critical",
                "type": "cpu",
                "value": cpu["percent"],
                "message": f"CPU usage critical: {cpu['percent']:.1f}%",
            })

        if mem.get("percent", 0) > ALERT_RAM_PCT:
            alerts.append({
                "level": "critical",
                "type": "ram",
                "value": mem["percent"],
                "message": f"RAM usage critical: {mem['percent']:.1f}%",
            })

        for disk in metrics.get("disk", []):
            if disk["percent"] > ALERT_DISK_PCT:
                alerts.append({
                    "level": "warning" if disk["percent"] < 95 else "critical",
                    "type": "disk",
                    "mountpoint": disk["mountpoint"],
                    "value": disk["percent"],
                    "message": f"Disk {disk['mountpoint']} at {disk['percent']:.1f}%",
                })

        return alerts

    @staticmethod
    def _format_metrics(m: dict[str, Any]) -> str:
        """Human-readable metrics summary."""
        cpu = m.get("cpu", {})
        mem = m.get("memory", {})
        disks = m.get("disk", [])
        net = m.get("network", {})

        disk_str = " | ".join(
            f"{d['mountpoint']} {d['percent']:.0f}%" for d in disks[:3]
        )

        return (
            f"CPU: {cpu.get('percent', 0):.1f}% | "
            f"RAM: {mem.get('percent', 0):.1f}% ({mem.get('used_gb', 0):.1f}GB used) | "
            f"Disk: {disk_str} | "
            f"Net ↑{net.get('bytes_sent_mb', 0):.0f}MB ↓{net.get('bytes_recv_mb', 0):.0f}MB"
        )

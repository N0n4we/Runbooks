# k3d monitoring stack

This directory contains raw Kubernetes manifests for a local k3d monitoring stack:

- Prometheus
- Loki
- Promtail
- Grafana
- Alertmanager

Apply:

```sh
kubectl apply -k .
```

Wait:

```sh
kubectl -n monitoring rollout status deploy/prometheus
kubectl -n monitoring rollout status deploy/loki
kubectl -n monitoring rollout status deploy/grafana
kubectl -n monitoring rollout status deploy/alertmanager
kubectl -n monitoring rollout status ds/promtail
```

Local access:

```sh
kubectl -n monitoring port-forward svc/grafana 3000:3000
kubectl -n monitoring port-forward svc/prometheus 9090:9090
kubectl -n monitoring port-forward svc/alertmanager 9093:9093
kubectl -n monitoring port-forward svc/loki 3100:3100
```

Grafana is available at <http://localhost:3000>.

- User: `admin`
- Password: `admin`
- Provisioned data sources: `Prometheus`, `Loki`, `Alertmanager`

Prometheus includes a `Watchdog` smoke-test alert so Alertmanager should show an active alert shortly after the stack is ready.

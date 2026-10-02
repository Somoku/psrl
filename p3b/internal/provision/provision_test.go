package provision_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"psrl.dev/sandboxd/internal/provision"
)

// -- agentenv -----------------------------------------------------------------

func TestAgentEnvGatewayPass(t *testing.T) {
	srv := okServer(t)
	r := provision.CheckAgentEnvGateway(context.Background(), srv.URL, "")
	if r.Err != nil {
		t.Fatalf("expected pass, got: %v", r.Err)
	}
}

func TestAgentEnvGatewayDown(t *testing.T) {
	r := provision.CheckAgentEnvGateway(context.Background(), "http://127.0.0.1:1", "")
	if r.Err == nil {
		t.Fatal("expected error for unreachable gateway")
	}
}

func TestAgentEnvGatewayAuthFail(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	r := provision.CheckAgentEnvGateway(context.Background(), srv.URL, "wrong-key")
	if r.Err == nil {
		t.Fatal("expected error for auth failure")
	}
	if !strings.Contains(r.Err.Error(), "401") {
		t.Errorf("error should mention HTTP 401, got: %v", r.Err)
	}
}

func TestAgentEnvNodePass(t *testing.T) {
	srv := okServer(t)
	r := provision.CheckAgentEnvNode(context.Background(), "n1", srv.URL, "")
	if r.Err != nil {
		t.Fatalf("expected pass, got: %v", r.Err)
	}
}

func TestAgentEnvNodeDown(t *testing.T) {
	r := provision.CheckAgentEnvNode(context.Background(), "n1", "http://127.0.0.1:1", "")
	if r.Err == nil {
		t.Fatal("expected error for unreachable node")
	}
	if !strings.Contains(r.Err.Error(), "n1") {
		t.Errorf("error should name the node id, got: %v", r.Err)
	}
}

// -- opensandbox --------------------------------------------------------------

func TestOpenSandboxGatewayPass(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// /health and /v1/sandboxes?pageSize=1 both return 200.
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"sandboxes":[]}`))
	}))
	defer srv.Close()
	r := provision.CheckOpenSandboxGateway(context.Background(), srv.URL, "key-1")
	if r.Err != nil {
		t.Fatalf("expected pass, got: %v", r.Err)
	}
}

func TestOpenSandboxGatewayDown(t *testing.T) {
	r := provision.CheckOpenSandboxGateway(context.Background(), "http://127.0.0.1:1", "")
	if r.Err == nil {
		t.Fatal("expected error for unreachable server")
	}
	// Error should mention installation instructions.
	if !strings.Contains(r.Err.Error(), "opensandbox-server") {
		t.Errorf("error should mention opensandbox-server install, got: %v", r.Err)
	}
}

func TestOpenSandboxGatewayAuthFail(t *testing.T) {
	call := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call++
		if call == 1 {
			// /health passes
			w.WriteHeader(http.StatusOK)
			return
		}
		// API key check fails
		w.WriteHeader(http.StatusForbidden)
	}))
	defer srv.Close()
	r := provision.CheckOpenSandboxGateway(context.Background(), srv.URL, "bad-key")
	if r.Err == nil {
		t.Fatal("expected error for forbidden API key")
	}
	if !strings.Contains(r.Err.Error(), "api_key") {
		t.Errorf("error should mention api_key, got: %v", r.Err)
	}
}

func TestOpenSandboxNodePass(t *testing.T) {
	srv := okServer(t)
	r := provision.CheckOpenSandboxNode(context.Background(), "n1", srv.URL, "")
	if r.Err != nil {
		t.Fatalf("expected pass, got: %v", r.Err)
	}
}

func TestOpenSandboxNodeDown(t *testing.T) {
	r := provision.CheckOpenSandboxNode(context.Background(), "n1", "http://127.0.0.1:1", "")
	if r.Err == nil {
		t.Fatal("expected error for unreachable node")
	}
	// Error should include install instructions.
	if !strings.Contains(r.Err.Error(), "pip install") {
		t.Errorf("error should include install command, got: %v", r.Err)
	}
}

// -- cubesandbox --------------------------------------------------------------

func TestCubeSandboxGatewayPass(t *testing.T) {
	srv := okServer(t)
	r := provision.CheckCubeSandboxGateway(context.Background(), srv.URL, "")
	if r.Err != nil {
		t.Fatalf("expected pass, got: %v", r.Err)
	}
}

func TestCubeSandboxGatewayDown(t *testing.T) {
	r := provision.CheckCubeSandboxGateway(context.Background(), "http://127.0.0.1:1", "")
	if r.Err == nil {
		t.Fatal("expected error for unreachable gateway")
	}
}

func TestCubeSandboxNodePass(t *testing.T) {
	srv := okServer(t)
	r := provision.CheckCubeSandboxNode(context.Background(), "n1", srv.URL, "")
	if r.Err != nil {
		t.Fatalf("expected pass, got: %v", r.Err)
	}
}

func TestCubeSandboxNodeDown(t *testing.T) {
	r := provision.CheckCubeSandboxNode(context.Background(), "n1", "http://127.0.0.1:1", "")
	if r.Err == nil {
		t.Fatal("expected error for unreachable cubelet")
	}
	if !strings.Contains(r.Err.Error(), "n1") {
		t.Errorf("error should name the node id, got: %v", r.Err)
	}
}

// -- result label -------------------------------------------------------------

func TestResultNameIsSet(t *testing.T) {
	srv := okServer(t)
	r := provision.CheckCubeSandboxNode(context.Background(), "my-node", srv.URL, "")
	if r.Name == "" {
		t.Fatal("result Name should not be empty")
	}
	if !strings.Contains(r.Name, "my-node") {
		t.Errorf("result Name should contain the node id, got %q", r.Name)
	}
}

// -- hint messages carry actionable text --------------------------------------

func TestHintCarriesActionableText(t *testing.T) {
	// A down node should give a "To fix:" hint in the error text so the
	// operator knows what to do without consulting the docs.
	r := provision.CheckOpenSandboxNode(context.Background(), "n1", "http://127.0.0.1:1", "")
	if r.Err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(r.Err.Error(), "To fix:") {
		t.Errorf("error should carry 'To fix:' hint, got: %v", r.Err)
	}
}

// -- helpers ------------------------------------------------------------------

func okServer(t *testing.T) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	}))
	t.Cleanup(srv.Close)
	return srv
}

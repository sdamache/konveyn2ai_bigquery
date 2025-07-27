"""
Docker-related test fixtures for integration testing with containers.
"""

import os
import subprocess
import time

import docker
import pytest
import requests


@pytest.fixture(scope="session")
def docker_client():
    """Docker client for container management."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except docker.errors.DockerException:
        pytest.skip("Docker is not available")


@pytest.fixture(scope="session")
def docker_compose_file():
    """Path to docker-compose file for testing."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "docker-compose.yml")


@pytest.fixture(scope="session")
def test_services_config():
    """Configuration for test services."""
    return {
        "svami": {
            "name": "konveyn2ai_svami_1",
            "port": 8080,
            "health_endpoint": "/health",
            "timeout": 15,
        },
        "janapada": {
            "name": "konveyn2ai_janapada_1",
            "port": 8081,
            "health_endpoint": "/health",
            "timeout": 15,
        },
        "amatya": {
            "name": "konveyn2ai_amatya_1",
            "port": 8082,
            "health_endpoint": "/health",
            "timeout": 15,
        },
    }


@pytest.fixture(scope="session")
def docker_services(docker_client, docker_compose_file, test_services_config):
    """Start Docker services for integration testing."""

    # Check if docker-compose is available
    try:
        subprocess.run(["docker-compose", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("docker-compose is not available")

    project_dir = os.path.dirname(docker_compose_file)

    # Start services
    subprocess.run(
        ["docker-compose", "-f", docker_compose_file, "up", "-d"],
        cwd=project_dir,
        check=True,
    )

    # Wait for services to be healthy
    for _service_name, config in test_services_config.items():
        wait_for_service_health(
            port=config["port"],
            endpoint=config["health_endpoint"],
            timeout=config["timeout"],
        )

    yield test_services_config

    # Cleanup - stop services
    subprocess.run(
        ["docker-compose", "-f", docker_compose_file, "down"], cwd=project_dir
    )


def wait_for_service_health(port: int, endpoint: str, timeout: int = 15):
    """Wait for a service to become healthy."""

    url = f"http://localhost:{port}{endpoint}"
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") in ["healthy", "degraded"]:
                    return True
        except (requests.ConnectionError, requests.Timeout):
            pass

        time.sleep(1)

    raise TimeoutError(
        f"Service at {url} did not become healthy within {timeout} seconds"
    )


@pytest.fixture
def isolated_containers(docker_client):
    """Create isolated containers for testing individual services."""

    containers = {}

    def create_container(
        image_name: str, service_name: str, port: int, env_vars: dict[str, str] = None
    ):
        """Create and start a container for testing."""

        container = docker_client.containers.run(
            image=image_name,
            name=f"test_{service_name}_{int(time.time())}",
            ports={f"{port}/tcp": port},
            environment=env_vars or {},
            detach=True,
            remove=True,  # Auto-remove when stopped
        )

        containers[service_name] = container

        # Wait for container to be ready
        wait_for_service_health(port, "/health")

        return container

    yield create_container

    # Cleanup containers
    for container in containers.values():
        try:
            container.stop(timeout=10)
        except Exception:
            container.kill()


@pytest.fixture
def mock_external_dependencies():
    """Mock external dependencies for Docker testing."""

    return {
        "google_credentials": {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": "-----BEGIN PRIVATE KEY-----\nTEST\n-----END PRIVATE KEY-----\n",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "123456789",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        },
        "environment_variables": {
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_API_KEY": "test-api-key",
            "INDEX_ENDPOINT_ID": "test-endpoint-123",
            "INDEX_ID": "test-index-456",
        },
    }


@pytest.fixture(scope="session")
def docker_network(docker_client):
    """Create isolated Docker network for testing."""

    network_name = f"test_network_{int(time.time())}"

    network = docker_client.networks.create(name=network_name, driver="bridge")

    yield network

    # Cleanup
    network.remove()


@pytest.fixture
def service_containers(docker_client, docker_network, mock_external_dependencies):
    """Create individual service containers with proper networking."""

    containers = []

    def start_service(
        service_name: str, image_tag: str, port: int, extra_env: dict[str, str] = None
    ):
        """Start a service container with proper configuration."""

        env_vars = {
            **mock_external_dependencies["environment_variables"],
            **(extra_env or {}),
        }

        container = docker_client.containers.run(
            image=f"konveyn2ai/{service_name}:{image_tag}",
            name=f"test_{service_name}_{int(time.time())}",
            network=docker_network.name,
            environment=env_vars,
            ports={f"{port}/tcp": port},
            detach=True,
        )

        containers.append(container)

        # Wait for service to be ready
        wait_for_service_health(port, "/health")

        return container

    yield start_service

    # Cleanup
    for container in containers:
        try:
            container.stop(timeout=10)
            container.remove()
        except Exception:
            try:
                container.kill()
                container.remove()
            except Exception:
                pass


class DockerTestHelper:
    """Helper class for Docker-based testing."""

    @staticmethod
    def build_test_images(docker_client, project_root: str):
        """Build Docker images for testing."""

        images = {}

        # Build each service image
        services = ["svami", "janapada", "amatya"]

        for service in services:
            dockerfile_path = os.path.join(project_root, f"Dockerfile.{service}")

            if os.path.exists(dockerfile_path):
                image, logs = docker_client.images.build(
                    path=project_root,
                    dockerfile=f"Dockerfile.{service}",
                    tag=f"konveyn2ai/{service}:test",
                )
                images[service] = image

        return images

    @staticmethod
    def get_container_logs(container, tail: int = 100):
        """Get logs from a container."""
        return container.logs(tail=tail).decode("utf-8")

    @staticmethod
    def exec_command_in_container(container, command: list[str]):
        """Execute a command inside a container."""
        result = container.exec_run(command)
        return result.exit_code, result.output.decode("utf-8")

    @staticmethod
    def wait_for_container_log(container, expected_text: str, timeout: int = 15):
        """Wait for specific text to appear in container logs."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            logs = DockerTestHelper.get_container_logs(container)
            if expected_text in logs:
                return True
            time.sleep(1)

        return False


@pytest.fixture
def docker_helper():
    """Provide Docker test helper."""
    return DockerTestHelper


@pytest.fixture
def container_health_checker():
    """Fixture for checking container health."""

    def check_health(container_name: str, port: int, endpoint: str = "/health"):
        """Check if a container is healthy."""
        try:
            url = f"http://localhost:{port}{endpoint}"
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                return data.get("status") == "healthy"
        except Exception:
            pass

        return False

    return check_health


@pytest.fixture
def docker_compose_project():
    """Manage docker-compose project lifecycle."""

    class DockerComposeProject:
        def __init__(self, compose_file: str):
            self.compose_file = compose_file
            self.project_dir = os.path.dirname(compose_file)

        def up(self, services: list[str] = None):
            """Start services."""
            cmd = ["docker-compose", "-f", self.compose_file, "up", "-d"]
            if services:
                cmd.extend(services)

            subprocess.run(cmd, cwd=self.project_dir, check=True)

        def down(self):
            """Stop services."""
            subprocess.run(
                ["docker-compose", "-f", self.compose_file, "down"],
                cwd=self.project_dir,
            )

        def logs(self, service: str = None):
            """Get logs from services."""
            cmd = ["docker-compose", "-f", self.compose_file, "logs"]
            if service:
                cmd.append(service)

            result = subprocess.run(
                cmd, cwd=self.project_dir, capture_output=True, text=True
            )
            return result.stdout

        def ps(self):
            """List running services."""
            result = subprocess.run(
                ["docker-compose", "-f", self.compose_file, "ps"],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
            )
            return result.stdout

    return DockerComposeProject


# Skip markers for Docker tests
def pytest_configure(config):
    """Configure Docker-specific markers."""
    config.addinivalue_line("markers", "docker: Tests requiring Docker")
    config.addinivalue_line(
        "markers", "docker_integration: Integration tests with Docker containers"
    )


def pytest_collection_modifyitems(config, items):
    """Skip Docker tests if Docker is not available."""

    try:
        docker.from_env().ping()
        docker_available = True
    except Exception:
        docker_available = False

    if not docker_available:
        skip_docker = pytest.mark.skip(reason="Docker is not available")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_docker)

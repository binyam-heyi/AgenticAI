"""
MCP Configuration Module
========================
Provides configuration for different MCP (Model Context Protocol) servers
including MySQL database and file system access.
"""

import os
import sys
import shutil
import site
from pathlib import Path
from autogen_ext.tools.mcp import StdioServerParams, McpWorkbench
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MCPConfig:
    """
    Configuration class for MCP (Model Context Protocol) servers.
    Provides methods to create workbenches for different tools like MySQL and file system.
    """

    @staticmethod
    def get_MySQL_ServerMCP():
        """
        Create and return an MCP workbench for MySQL database operations.
        This allows the AI assistant to query and interact with the MySQL database.

        Uses dynamic path resolution to find the 'uv' command and site-packages directory.
        Database credentials are loaded from environment variables for security.

        Returns:
            McpWorkbench: Configured MySQL MCP workbench

        Raises:
            FileNotFoundError: If 'uv' command cannot be found
            ValueError: If required environment variables are missing
        """

        # Find uv command dynamically
        uv_path = shutil.which("uv")

        # If not in PATH, try to find it relative to Python installation
        if not uv_path:
            python_dir = Path(sys.executable).parent
            possible_locations = [
                python_dir / "Scripts" / "uv.exe",  # Windows
                python_dir / "Scripts" / "uv",  # Windows (no .exe)
                python_dir / "bin" / "uv",  # Linux/Mac
            ]

            for location in possible_locations:
                if location.exists():
                    uv_path = str(location)
                    break

        if not uv_path:
            raise FileNotFoundError(
                "❌ 'uv' command not found. Please install it: pip install uv"
            )


        site_packages = site.getsitepackages()[0]
        site_packages_str = str(Path(site_packages)).replace("\\", "/")

        print(f"✓ Using uv: {uv_path}")
        print(f"✓ Using site-packages: {site_packages_str}")

        #  Load MySQL configuration from environment variables
        mysql_config = {
            "MYSQL_HOST": os.getenv("MYSQL_HOST", "localhost"),
            "MYSQL_PORT": os.getenv("MYSQL_PORT", "3306"),
            "MYSQL_USER": os.getenv("MYSQL_USER", "root"),
            "MYSQL_PASSWORD": os.getenv("MYSQL_PASSWORD", "1234"),
            "MYSQL_DATABASE": os.getenv("MYSQL_DATABASE", "school_db")
        }

        # Validate that critical configuration exists
        if not mysql_config["MYSQL_DATABASE"]:
            raise ValueError("❌ MYSQL_DATABASE environment variable is required")

        print(f"📊 Connecting to database: {mysql_config['MYSQL_DATABASE']}\n")

        # Configure MySQL MCP server parameters
        mysql_server_params = StdioServerParams(
            command=uv_path,  # ✅ Dynamic path
            args=[
                "--directory",
                site_packages_str,  # ✅ Dynamic path
                "run",
                "mysql_mcp_server"
            ],
            env=mysql_config,
            read_timeout_seconds=60
        )

        # Create and return the MCP workbench
        mysql_workbench = McpWorkbench(mysql_server_params)
        return mysql_workbench

    @staticmethod
    def get_FileSystem_ServerMCP(directory=None):
        """
        Create and return an MCP workbench for file system operations.
        This allows the AI assistant to read/write files in the specified directory.

        Args:
            directory (str, optional): Directory path to grant access to.
                                      If None, uses the current script's directory.

        Returns:
            McpWorkbench: Configured file system MCP workbench
        """
        # ✅ Use provided directory or default to script's directory
        if directory is None:
            directory = Path(__file__).parent.resolve()

        directory_str = str(Path(directory)).replace("\\", "/")

        print(f"📁 File system access granted to: {directory_str}\n")

        # Configure file system MCP server parameters
        file_server_params = StdioServerParams(
            command="npx",
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                directory_str
            ],
            read_timeout_seconds=60
        )

        # Create and return the MCP workbench
        file_workbench = McpWorkbench(file_server_params)
        return file_workbench

    @staticmethod
    def get_RestApi_ServerMCP(base_url=None, accept_header=None, **env_vars):
        """
        Create and return an MCP workbench configured to run the dkmaker-mcp-rest-api.
        This allows the AI assistant to interact with a RESTful API.

        Args:
            base_url (str, optional): The base URL for the REST API.
                                      Defaults to the mock URL if None.
            accept_header (str, optional): The value for the 'Accept' header.
                                           Defaults to 'application/json' if None.
            **env_vars: Additional environment variables to pass (e.g., for headers like Authorization).

        Returns:
            McpWorkbench: Configured REST API MCP workbench
        """
        # 1. Define Defaults
        default_base_url = "https://fake-json-api.mock.beeceptor.com/users"
        default_accept = "application/json"

        # 2. Set Final Values
        rest_base_url = base_url if base_url is not None else default_base_url
        header_accept = accept_header if accept_header is not None else default_accept

        print(f"🔗 REST API access configured for: {rest_base_url}\n")

        # 3. Build Environment Variables (env)
        env = {
            "REST_BASE_URL": rest_base_url,
            "HEADER_Accept": header_accept,
            **env_vars  # Include any additional headers/env vars passed in
        }

        # 4. Configure REST API MCP server parameters
        rest_server_params = StdioServerParams(
            # NOTE: Assuming the dkmaker-mcp-rest-api is globally installed for the user
            #       running the script, similar to how the 'npx' command is used.
            command="node",
            args=[
                "PATH TO -> /dkmaker-mcp-rest-api/build/index.js" # Change to your local path
            ],
            env=env,
            read_timeout_seconds=60
        )

        # 5. Create and return the MCP workbench
        rest_workbench = McpWorkbench(rest_server_params)
        return rest_workbench

    @staticmethod
    def get_combined_workbenches():
        """
        Get both MySQL and file system workbenches together.
        Useful when you need multiple capabilities in one assistant.

        Returns:
            tuple: (mysql_workbench, file_workbench)
        """
        mysql_wb = MCPConfig.get_MySQL_ServerMCP()
        file_wb = MCPConfig.get_FileSystem_ServerMCP()
        return mysql_wb, file_wb
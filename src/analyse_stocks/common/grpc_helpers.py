from __future__ import annotations

import logging

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc


def create_server() -> tuple[grpc.aio.Server, health.HealthServicer]:
    server = grpc.aio.server()
    health_servicer = health.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    return server, health_servicer


async def mark_serving(health_servicer: health.HealthServicer, service_names: list[str]) -> None:
    for service_name in service_names:
        health_servicer.set(service_name, health_pb2.HealthCheckResponse.SERVING)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)


async def run_server(server: grpc.aio.Server, bind_address: str, service_name: str) -> None:
    logging.info("Starting gRPC server on %s", bind_address)
    server.add_insecure_port(bind_address)
    await server.start()
    logging.info("%s is ready", service_name)
    await server.wait_for_termination()

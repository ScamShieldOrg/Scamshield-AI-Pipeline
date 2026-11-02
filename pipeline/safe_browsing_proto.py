"""Decode the Safe Browsing v5 urls.search Protocol Buffers response.

Only the response fields used by Tier 1 are described here. Field numbers follow
SearchUrlsResponse, ThreatUrl, and ThreatType in the Safe Browsing v5 schema.
Unknown protobuf fields remain forward compatible.
"""

from google.protobuf import descriptor_pb2, descriptor_pool, duration_pb2, message_factory
from google.protobuf.message import DecodeError


THREAT_TYPES = {
    1: "MALWARE",
    2: "SOCIAL_ENGINEERING",
    3: "UNWANTED_SOFTWARE",
    4: "POTENTIALLY_HARMFUL_APPLICATION",
}


def _search_urls_response_class():
    schema = descriptor_pb2.FileDescriptorProto()
    schema.name = "scamshield/safe_browsing_search_urls.proto"
    schema.package = "google.security.safebrowsing.v5"
    schema.syntax = "proto3"
    schema.dependency.append("google/protobuf/duration.proto")

    threat = schema.message_type.add()
    threat.name = "ThreatUrl"
    url_field = threat.field.add()
    url_field.name, url_field.number, url_field.label, url_field.type = (
        "url", 1, descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
    )
    types_field = threat.field.add()
    types_field.name, types_field.number, types_field.label, types_field.type = (
        "threat_types", 2, descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        descriptor_pb2.FieldDescriptorProto.TYPE_INT32,
    )

    result = schema.message_type.add()
    result.name = "SearchUrlsResponse"
    threats_field = result.field.add()
    threats_field.name, threats_field.number, threats_field.label, threats_field.type = (
        "threats", 1, descriptor_pb2.FieldDescriptorProto.LABEL_REPEATED,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
    )
    threats_field.type_name = ".google.security.safebrowsing.v5.ThreatUrl"
    duration_field = result.field.add()
    duration_field.name, duration_field.number, duration_field.label, duration_field.type = (
        "cache_duration", 2, descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
    )
    duration_field.type_name = ".google.protobuf.Duration"

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(duration_pb2.DESCRIPTOR.serialized_pb)
    descriptor = pool.Add(schema)
    return message_factory.GetMessageClass(descriptor.message_types_by_name["SearchUrlsResponse"])


SearchUrlsResponse = _search_urls_response_class()


def decode_search_urls_response(body: bytes) -> tuple[list[str], float]:
    """Return matched threat names and the server's cache duration in seconds."""
    response = SearchUrlsResponse()
    try:
        response.ParseFromString(body)
    except DecodeError as exc:
        raise ValueError("Invalid Safe Browsing protobuf") from exc
    if not response.HasField("cache_duration"):
        raise ValueError("Safe Browsing protobuf lacks cache duration")
    duration = response.cache_duration
    if duration.seconds < 0 or not 0 <= duration.nanos < 1_000_000_000:
        raise ValueError("Invalid Safe Browsing cache duration")
    threat_types = set()
    for threat in response.threats:
        if not threat.url or not threat.threat_types:
            raise ValueError("Invalid Safe Browsing threat entry")
        for kind in threat.threat_types:
            if kind == 0:
                raise ValueError("Unspecified Safe Browsing threat type")
            threat_types.add(THREAT_TYPES.get(kind, f"UNKNOWN_THREAT_TYPE_{kind}"))
    return sorted(threat_types), duration.seconds + duration.nanos / 1_000_000_000

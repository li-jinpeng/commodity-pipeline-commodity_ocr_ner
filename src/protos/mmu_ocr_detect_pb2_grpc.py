# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import mmu_ocr_detect_pb2 as mmu__ocr__detect__pb2


class MmuOcrDetectServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.OcrDetect = channel.unary_unary(
                '/mmu.image.MmuOcrDetectService/OcrDetect',
                request_serializer=mmu__ocr__detect__pb2.OcrDetectRequest.SerializeToString,
                response_deserializer=mmu__ocr__detect__pb2.OcrDetectResponse.FromString,
                )
        self.GetPhotoCoverOcr = channel.unary_unary(
                '/mmu.image.MmuOcrDetectService/GetPhotoCoverOcr',
                request_serializer=mmu__ocr__detect__pb2.PhotoCoverOcrRequest.SerializeToString,
                response_deserializer=mmu__ocr__detect__pb2.PhotoCoverOcrResponse.FromString,
                )
        self.GetPhotoOcr = channel.unary_unary(
                '/mmu.image.MmuOcrDetectService/GetPhotoOcr',
                request_serializer=mmu__ocr__detect__pb2.PhotoCoverOcrRequest.SerializeToString,
                response_deserializer=mmu__ocr__detect__pb2.PhotoOcrResponse.FromString,
                )


class MmuOcrDetectServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def OcrDetect(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetPhotoCoverOcr(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetPhotoOcr(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_MmuOcrDetectServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'OcrDetect': grpc.unary_unary_rpc_method_handler(
                    servicer.OcrDetect,
                    request_deserializer=mmu__ocr__detect__pb2.OcrDetectRequest.FromString,
                    response_serializer=mmu__ocr__detect__pb2.OcrDetectResponse.SerializeToString,
            ),
            'GetPhotoCoverOcr': grpc.unary_unary_rpc_method_handler(
                    servicer.GetPhotoCoverOcr,
                    request_deserializer=mmu__ocr__detect__pb2.PhotoCoverOcrRequest.FromString,
                    response_serializer=mmu__ocr__detect__pb2.PhotoCoverOcrResponse.SerializeToString,
            ),
            'GetPhotoOcr': grpc.unary_unary_rpc_method_handler(
                    servicer.GetPhotoOcr,
                    request_deserializer=mmu__ocr__detect__pb2.PhotoCoverOcrRequest.FromString,
                    response_serializer=mmu__ocr__detect__pb2.PhotoOcrResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'mmu.image.MmuOcrDetectService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class MmuOcrDetectService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def OcrDetect(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mmu.image.MmuOcrDetectService/OcrDetect',
            mmu__ocr__detect__pb2.OcrDetectRequest.SerializeToString,
            mmu__ocr__detect__pb2.OcrDetectResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetPhotoCoverOcr(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mmu.image.MmuOcrDetectService/GetPhotoCoverOcr',
            mmu__ocr__detect__pb2.PhotoCoverOcrRequest.SerializeToString,
            mmu__ocr__detect__pb2.PhotoCoverOcrResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetPhotoOcr(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/mmu.image.MmuOcrDetectService/GetPhotoOcr',
            mmu__ocr__detect__pb2.PhotoCoverOcrRequest.SerializeToString,
            mmu__ocr__detect__pb2.PhotoOcrResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

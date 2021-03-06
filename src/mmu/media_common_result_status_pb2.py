# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mmu/media_common_result_status.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='mmu/media_common_result_status.proto',
  package='mmu.common',
  syntax='proto3',
  serialized_options=b'\n\027com.kuaishou.mmu.commonB\006Result',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n$mmu/media_common_result_status.proto\x12\nmmu.common\"A\n\x0cResultStatus\x12$\n\x04\x63ode\x18\x01 \x01(\x0e\x32\x16.mmu.common.ResultCode\x12\x0b\n\x03msg\x18\x02 \x01(\t\"K\n\x0eStringResponse\x12(\n\x06status\x18\x01 \x01(\x0b\x32\x18.mmu.common.ResultStatus\x12\x0f\n\x07resullt\x18\x02 \x01(\t\"*\n\x18RaiseSystemSignalRequest\x12\x0e\n\x06signal\x18\x01 \x01(\r*\xb5\x05\n\nResultCode\x12\n\n\x06UNKOWN\x10\x00\x12\n\n\x06SUCESS\x10\x01\x12\x0f\n\x0bINPUT_EMPTY\x10\x02\x12\t\n\x05\x45RROR\x10\x03\x12\x1d\n\x19\x42\x41IDU_AUDIO_SERVICE_ERROR\x10\x04\x12\x10\n\x0cOUTPUT_EMPTY\x10\x05\x12\x12\n\x0ePARTAL_SUCCESS\x10\x06\x12\x14\n\x10NO_FACE_DETECTED\x10\x07\x12\x13\n\x0fNO_FACE_MATCHED\x10\x08\x12\x19\n\x15ILLEGAL_FACE_DETECTED\x10\t\x12\n\n\x06\x46\x41ILED\x10\n\x12\x14\n\x10\x42LOB_STORE_ERROR\x10\x0b\x12\x0b\n\x07ILLEGAL\x10\x0c\x12\x13\n\x0fOVER_RATE_LIMIT\x10\r\x12\n\n\x06REJECT\x10\x0e\x12\x17\n\x13\x45XCEED_MAXIMUM_SIZE\x10\x0f\x12\r\n\tBAD_ANGLE\x10\x10\x12\x0f\n\x0b\x42\x41\x44_CLARITY\x10\x11\x12\x0b\n\x07TIMEOUT\x10\x12\x12\x0f\n\x0bPARAM_ERROR\x10\x13\x12\x0e\n\nPROCESSING\x10\x14\x12\x16\n\x12TRANSCODING_FAILED\x10\x15\x12\x1d\n\x19SPEECH_RECOGNITION_FAILED\x10\x16\x12\r\n\tVAD_EMPTY\x10\x17\x12\x0f\n\x0bVAD_TIMEOUT\x10\x18\x12\x1e\n\x1aSPEECH_RECOGNITION_TIMEOUT\x10\x19\x12\x13\n\x0fGET_DATA_FAILED\x10\x1a\x12\x14\n\x10KEY_FRAME_FAILED\x10\x1b\x12\x1a\n\x16\x41UIDO_CLIP_INPUT_ERROR\x10\x1c\x12\"\n\x1e\x41UDIO_CLIP_OUT_BOUNDS_OF_START\x10\x1d\x12 \n\x1c\x41UDIO_CLIP_OUT_BOUNDS_OF_END\x10\x1e\x12\x1a\n\x16\x41UDIO_CLIP_OTHER_ERROR\x10\x1f\x12\x11\n\rAUDIO_NOVOCAL\x10 B!\n\x17\x63om.kuaishou.mmu.commonB\x06Resultb\x06proto3'
)

_RESULTCODE = _descriptor.EnumDescriptor(
  name='ResultCode',
  full_name='mmu.common.ResultCode',
  filename=None,
  file=DESCRIPTOR,
  create_key=_descriptor._internal_create_key,
  values=[
    _descriptor.EnumValueDescriptor(
      name='UNKOWN', index=0, number=0,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SUCESS', index=1, number=1,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='INPUT_EMPTY', index=2, number=2,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ERROR', index=3, number=3,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BAIDU_AUDIO_SERVICE_ERROR', index=4, number=4,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='OUTPUT_EMPTY', index=5, number=5,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PARTAL_SUCCESS', index=6, number=6,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='NO_FACE_DETECTED', index=7, number=7,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='NO_FACE_MATCHED', index=8, number=8,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ILLEGAL_FACE_DETECTED', index=9, number=9,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='FAILED', index=10, number=10,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BLOB_STORE_ERROR', index=11, number=11,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='ILLEGAL', index=12, number=12,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='OVER_RATE_LIMIT', index=13, number=13,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='REJECT', index=14, number=14,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='EXCEED_MAXIMUM_SIZE', index=15, number=15,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BAD_ANGLE', index=16, number=16,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='BAD_CLARITY', index=17, number=17,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TIMEOUT', index=18, number=18,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PARAM_ERROR', index=19, number=19,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='PROCESSING', index=20, number=20,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='TRANSCODING_FAILED', index=21, number=21,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SPEECH_RECOGNITION_FAILED', index=22, number=22,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='VAD_EMPTY', index=23, number=23,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='VAD_TIMEOUT', index=24, number=24,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='SPEECH_RECOGNITION_TIMEOUT', index=25, number=25,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='GET_DATA_FAILED', index=26, number=26,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='KEY_FRAME_FAILED', index=27, number=27,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='AUIDO_CLIP_INPUT_ERROR', index=28, number=28,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='AUDIO_CLIP_OUT_BOUNDS_OF_START', index=29, number=29,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='AUDIO_CLIP_OUT_BOUNDS_OF_END', index=30, number=30,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='AUDIO_CLIP_OTHER_ERROR', index=31, number=31,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
    _descriptor.EnumValueDescriptor(
      name='AUDIO_NOVOCAL', index=32, number=32,
      serialized_options=None,
      type=None,
      create_key=_descriptor._internal_create_key),
  ],
  containing_type=None,
  serialized_options=None,
  serialized_start=241,
  serialized_end=934,
)
_sym_db.RegisterEnumDescriptor(_RESULTCODE)

ResultCode = enum_type_wrapper.EnumTypeWrapper(_RESULTCODE)
UNKOWN = 0
SUCESS = 1
INPUT_EMPTY = 2
ERROR = 3
BAIDU_AUDIO_SERVICE_ERROR = 4
OUTPUT_EMPTY = 5
PARTAL_SUCCESS = 6
NO_FACE_DETECTED = 7
NO_FACE_MATCHED = 8
ILLEGAL_FACE_DETECTED = 9
FAILED = 10
BLOB_STORE_ERROR = 11
ILLEGAL = 12
OVER_RATE_LIMIT = 13
REJECT = 14
EXCEED_MAXIMUM_SIZE = 15
BAD_ANGLE = 16
BAD_CLARITY = 17
TIMEOUT = 18
PARAM_ERROR = 19
PROCESSING = 20
TRANSCODING_FAILED = 21
SPEECH_RECOGNITION_FAILED = 22
VAD_EMPTY = 23
VAD_TIMEOUT = 24
SPEECH_RECOGNITION_TIMEOUT = 25
GET_DATA_FAILED = 26
KEY_FRAME_FAILED = 27
AUIDO_CLIP_INPUT_ERROR = 28
AUDIO_CLIP_OUT_BOUNDS_OF_START = 29
AUDIO_CLIP_OUT_BOUNDS_OF_END = 30
AUDIO_CLIP_OTHER_ERROR = 31
AUDIO_NOVOCAL = 32



_RESULTSTATUS = _descriptor.Descriptor(
  name='ResultStatus',
  full_name='mmu.common.ResultStatus',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='code', full_name='mmu.common.ResultStatus.code', index=0,
      number=1, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='msg', full_name='mmu.common.ResultStatus.msg', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=52,
  serialized_end=117,
)


_STRINGRESPONSE = _descriptor.Descriptor(
  name='StringResponse',
  full_name='mmu.common.StringResponse',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='status', full_name='mmu.common.StringResponse.status', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='resullt', full_name='mmu.common.StringResponse.resullt', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=119,
  serialized_end=194,
)


_RAISESYSTEMSIGNALREQUEST = _descriptor.Descriptor(
  name='RaiseSystemSignalRequest',
  full_name='mmu.common.RaiseSystemSignalRequest',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='signal', full_name='mmu.common.RaiseSystemSignalRequest.signal', index=0,
      number=1, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=196,
  serialized_end=238,
)

_RESULTSTATUS.fields_by_name['code'].enum_type = _RESULTCODE
_STRINGRESPONSE.fields_by_name['status'].message_type = _RESULTSTATUS
DESCRIPTOR.message_types_by_name['ResultStatus'] = _RESULTSTATUS
DESCRIPTOR.message_types_by_name['StringResponse'] = _STRINGRESPONSE
DESCRIPTOR.message_types_by_name['RaiseSystemSignalRequest'] = _RAISESYSTEMSIGNALREQUEST
DESCRIPTOR.enum_types_by_name['ResultCode'] = _RESULTCODE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ResultStatus = _reflection.GeneratedProtocolMessageType('ResultStatus', (_message.Message,), {
  'DESCRIPTOR' : _RESULTSTATUS,
  '__module__' : 'mmu.media_common_result_status_pb2'
  # @@protoc_insertion_point(class_scope:mmu.common.ResultStatus)
  })
_sym_db.RegisterMessage(ResultStatus)

StringResponse = _reflection.GeneratedProtocolMessageType('StringResponse', (_message.Message,), {
  'DESCRIPTOR' : _STRINGRESPONSE,
  '__module__' : 'mmu.media_common_result_status_pb2'
  # @@protoc_insertion_point(class_scope:mmu.common.StringResponse)
  })
_sym_db.RegisterMessage(StringResponse)

RaiseSystemSignalRequest = _reflection.GeneratedProtocolMessageType('RaiseSystemSignalRequest', (_message.Message,), {
  'DESCRIPTOR' : _RAISESYSTEMSIGNALREQUEST,
  '__module__' : 'mmu.media_common_result_status_pb2'
  # @@protoc_insertion_point(class_scope:mmu.common.RaiseSystemSignalRequest)
  })
_sym_db.RegisterMessage(RaiseSystemSignalRequest)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)

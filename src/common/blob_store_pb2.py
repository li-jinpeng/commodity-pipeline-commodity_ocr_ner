# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: common/blob_store.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='common/blob_store.proto',
  package='',
  syntax='proto3',
  serialized_options=b'\n\034com.kuaishou.common.protobufB\016ProtoBlobStore',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n\x17\x63ommon/blob_store.proto\";\n\x11ProtoBlobStoreKey\x12\n\n\x02\x64\x62\x18\x01 \x01(\t\x12\r\n\x05table\x18\x02 \x01(\t\x12\x0b\n\x03key\x18\x03 \x01(\tB.\n\x1c\x63om.kuaishou.common.protobufB\x0eProtoBlobStoreb\x06proto3'
)




_PROTOBLOBSTOREKEY = _descriptor.Descriptor(
  name='ProtoBlobStoreKey',
  full_name='ProtoBlobStoreKey',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='db', full_name='ProtoBlobStoreKey.db', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='table', full_name='ProtoBlobStoreKey.table', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='key', full_name='ProtoBlobStoreKey.key', index=2,
      number=3, type=9, cpp_type=9, label=1,
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
  serialized_start=27,
  serialized_end=86,
)

DESCRIPTOR.message_types_by_name['ProtoBlobStoreKey'] = _PROTOBLOBSTOREKEY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ProtoBlobStoreKey = _reflection.GeneratedProtocolMessageType('ProtoBlobStoreKey', (_message.Message,), {
  'DESCRIPTOR' : _PROTOBLOBSTOREKEY,
  '__module__' : 'common.blob_store_pb2'
  # @@protoc_insertion_point(class_scope:ProtoBlobStoreKey)
  })
_sym_db.RegisterMessage(ProtoBlobStoreKey)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
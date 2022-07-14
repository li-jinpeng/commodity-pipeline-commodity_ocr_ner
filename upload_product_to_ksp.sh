#!/bin/bash
# liyong <liyong@kuaishou.com>
#  将./archive.sh生成的包（产品），上传到ksp产品库

function display_help() {
    echo ""
    echo "Usage: ./upload_product_to_ksp.sh -n <product_name> -f <product_file_path> [-m <message>]"
    echo ""
    echo "    必选参数:"
    echo "        -n product name which is registered in ksp."
    echo "        -f local product file, maybe a tar.gz file"
    echo "    可选参数:"
    echo "        -m upload message"
    echo ""
    echo ""
}

PRODUCT_NAME=""
PRODUCT_FILE=""
MESSAGE=""

for o in "$@"
do
    case "${o}" in
       -n)          shift; PRODUCT_NAME="${1}"; shift; ;;
       -f)          shift; PRODUCT_FILE="${1}"; shift; ;;
       -m)           shift; MESSAGE="${1}"; shift; ;;
       -h)          display_help && exit 0; ;;
       -*)              echo "Unknown option ${o}.  Try --help."; exit 1; ;;
    esac
done

if [[ -z "${PRODUCT_NAME}" ]]; then
    echo ""
    echo "ERROR: 请输入-n参数"
    display_help && exit 1;
fi

if [[ -z "${PRODUCT_FILE}" ]]; then
    echo ""
    echo "ERROR: 请输入-f参数"
    display_help && exit 1;
fi

if [[ ! -s ${PRODUCT_FILE} ]]; then
    echo ""
    echo "ERROR: PRODUCT_FILE不存在或文件大小为0!"
    display_help && exit 1;
fi

SUFFIX=${PRODUCT_FILE#${PRODUCT_NAME}}
SUFFIX=${SUFFIX##*-}
SUFFIX=${SUFFIX%.tar.gz}
echo "PRODUCT_NAME=${PRODUCT_NAME}"
echo "PRODUCT_FILE=${PRODUCT_FILE}"
echo "MESSAGE=${MESSAGE}"
echo "SUFFIX=${SUFFIX}"


if ! echo "${PRODUCT_FILE}" | egrep -q "^dist/${PRODUCT_NAME}" 2>/dev/null; then
    echo "ERROR: 产品名称(${PRODUCT_NAME})与产品文件名前缀(${PRODUCT_FILE%%.tar.gz})不匹配！"
    exit 1;
fi

echo "准备上传"
curl --verbose \
    "http://ksp.corp.kuaishou.com/api/product/products/${PRODUCT_NAME}/versions/upload/" \
    -H "host: ksp.corp.kuaishou.com" \
    -F "msg=\"${MESSAGE}\"" \
    -F "file=@${PRODUCT_FILE}" \
    -F "suffix=${SUFFIX}"
echo ""
echo "上传结束"

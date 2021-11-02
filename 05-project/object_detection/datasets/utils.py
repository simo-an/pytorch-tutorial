# 将 xml 格式 转化为 json 格式
def parse_xml_to_dict(xml):
    if len(xml) == 0:
        return {xml.tag: xml.text}
    
    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    
    return {xml.tag: result}

class_dict = {
    "crazing": 1,
    "inclusion": 2,
    "patches": 3,
    "pitted_surface": 4,
    "rolled-in_scale": 5,
    "scratches": 6,
}

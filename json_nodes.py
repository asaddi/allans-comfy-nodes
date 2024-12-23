# Copyright (c) 2024 Allan Saddi <allan@saddi.com>
import jmespath


class JSONExtractNumber:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("JSON",),
                "jmespath_query": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "((prompt.*)[?_meta.title=='Seed'].inputs.seed)[0]",
                    },
                ),
                "strict": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "default_value": ("FLOAT",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, jmespath_query):
        if not jmespath_query:
            return "required"
        return True

    TITLE = "JSON Extract Number"

    RETURN_TYPES = ("FLOAT,INT",)

    FUNCTION = "json_extract"

    CATEGORY = "private/json"

    def json_extract(
        self,
        json_input,
        jmespath_query: str,
        strict: bool,
        default_value: int | float,
    ):
        # TODO handle lists natively, compile expression, etc.
        value = jmespath.search(jmespath_query, json_input)
        if value is None:
            if strict:
                # TODO Maybe we should dump actual JSON to console too?
                raise ValueError("JMESPath query returned null")
            else:
                value = default_value
        # TODO Maybe a cast/coerce option?
        if not isinstance(value, (int, float)):
            print(f"{type(self).__name__}: Extracted value: {repr(value)}")
            raise ValueError("JMESPath query did not return a number")
        return (value,)


class JSONExtractString:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "json_input": ("JSON",),
                "jmespath_query": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "((prompt.*)[?_meta.title=='Positive Prompt'].inputs.text)[0]",
                    },
                ),
                "strict": (
                    "BOOLEAN",
                    {
                        "default": True,
                    },
                ),
                "default_value": ("STRING",),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, jmespath_query):
        if not jmespath_query:
            return "required"
        return True

    TITLE = "JSON Extract String"

    RETURN_TYPES = ("STRING",)

    FUNCTION = "json_extract"

    CATEGORY = "private/json"

    def json_extract(
        self, json_input, jmespath_query: str, strict: bool, default_value: str
    ):
        # TODO handle lists natively, compile expression, etc.
        value = jmespath.search(jmespath_query, json_input)
        if value is None:
            if strict:
                # TODO Maybe we should dump actual JSON to console too?
                raise ValueError("JMESPath query returned null")
            else:
                value = default_value
        # TODO Maybe a cast/coerce option?
        if not isinstance(value, str):
            print(f"{type(self).__name__}: Extracted value: {repr(value)}")
            raise ValueError("JMESPath query did not return a string")
        return (value,)


NODE_CLASS_MAPPINGS = {
    "JSONExtractNumber": JSONExtractNumber,
    "JSONExtractString": JSONExtractString,
}

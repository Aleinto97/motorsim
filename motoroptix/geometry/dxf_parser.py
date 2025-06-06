import ezdxf
from ezdxf.enums import TextEntityAlignment
from ezdxf.math import BoundingBox, Vec2
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)

class DXFParser:
    """
    Handles DXF file parsing, geometry extraction, and parameterization.
    """

    def __init__(self, dxf_filepath: str):
        self.filepath = dxf_filepath
        self.doc = None
        self.modelspace = None
        self._load_dxf()

    def _load_dxf(self):
        """Loads the DXF document and performs basic error checking."""
        try:
            self.doc = ezdxf.readfile(self.filepath)
            self.modelspace = self.doc.modelspace()
        except ezdxf.DXFError as e:
            logger.error(f"Failed to load DXF file '{self.filepath}': {e}")
            raise

    def get_all_entities(self) -> List[Any]:
        """Returns all entities in the modelspace."""
        if self.modelspace:
            return list(self.modelspace)
        return []

    def extract_parameters(self) -> Dict[str, Any]:
        """
        Extracts predefined parameters from DXF entities.
        This is a conceptual implementation. Real-world implementation
        would involve sophisticated geometry recognition.
        """
        parameters = {}
        if not self.modelspace:
            logger.warning("No modelspace loaded to extract parameters.")
            return parameters

        # Example: Try to find a circle assumed to be rotor outer boundary
        circles = self.modelspace.query("CIRCLE")
        if circles:
            # Find rotor circle by layer
            rotor_circle = next((c for c in circles if c.dxf.layer.upper() == 'ROTOR'), None)
            if rotor_circle:
                parameters["rotor_outer_radius"] = rotor_circle.dxf.radius
                parameters["rotor_center_x"] = rotor_circle.dxf.center.x
                parameters["rotor_center_y"] = rotor_circle.dxf.center.y
                logger.info(f"Detected rotor outer radius: {rotor_circle.dxf.radius}")
            else:
                logger.warning("Could not identify a rotor outer circle.")

            # Find stator circle by layer
            stator_circle = next((c for c in circles if c.dxf.layer.upper() in ['STATOR', 'STATOR_OUTER']), None)
            if stator_circle:
                parameters["stator_outer_radius"] = stator_circle.dxf.radius
                logger.info(f"Detected stator outer radius: {stator_circle.dxf.radius}")
            else:
                logger.warning("Could not identify a stator outer circle.")

        # More sophisticated logic required here for slots, poles, etc.
        # This would involve iterating polylines, lines, arcs and using geometric
        # processing (e.g., ezdxf.edgeminer, ezdxf.edgesmith) to identify features.
        # Example: Looking for text labels that might denote parameters
        for text_entity in self.modelspace.query("TEXT MTEXT"):
            content = text_entity.dxf.text.strip()
            if "=" in content:
                try:
                    key, value = content.split("=", 1)
                    key = key.strip().lower().replace(" ", "_")
                    parameters[key] = float(value.strip()) # Assume float for now
                    logger.info(f"Extracted parameter from text: {key}={value}")
                except ValueError:
                    pass # Not a simple key=value pair

        return parameters

    def create_dxf_from_parameters(self, parameters: Dict[str, Any], output_filepath: str):
        """
        Creates a simple DXF file based on extracted or defined parameters.
        This demonstrates the reverse process: parameterization to geometry.
        """
        new_doc = ezdxf.new("R2018")
        msp = new_doc.modelspace()

        if "rotor_outer_radius" in parameters and "rotor_center_x" in parameters and "rotor_center_y" in parameters:
            msp.add_circle(
                center=(parameters["rotor_center_x"], parameters["rotor_center_y"]),
                radius=parameters["rotor_outer_radius"],
                dxfattribs={'layer': 'ROTOR'}
            )
            msp.add_text("Rotor Outer", dxfattribs={
                'insert': (parameters["rotor_center_x"] + parameters["rotor_outer_radius"], parameters["rotor_center_y"]),
                'height': parameters["rotor_outer_radius"] / 10,
                'layer': 'LABELS'
            })

        if "stator_outer_radius" in parameters and "rotor_center_x" in parameters and "rotor_center_y" in parameters:
             msp.add_circle(
                center=(parameters["rotor_center_x"], parameters["rotor_center_y"]), # Assuming concentric
                radius=parameters["stator_outer_radius"],
                dxfattribs={'layer': 'STATOR'}
            )
             msp.add_text("Stator Outer", dxfattribs={
                'insert': (parameters["rotor_center_x"] + parameters["stator_outer_radius"], parameters["rotor_center_y"]),
                'height': parameters["stator_outer_radius"] / 10,
                'layer': 'LABELS'
            })

        # Add more logic here to draw slots, magnets, etc. based on parameters.
        # This would involve complex calculations using ezdxf.math module.

        new_doc.saveas(output_filepath)
        logger.info(f"DXF file created at '{output_filepath}' from parameters.")

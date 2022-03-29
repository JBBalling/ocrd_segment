from __future__ import absolute_import

import json

from shapely.geometry import Polygon
from PIL import Image, ImageDraw

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    polygon_from_points,
    xywh_from_polygon,
    MIME_TO_EXT
)
from ocrd_modelfactory import page_from_file
from ocrd import Processor

from .config import OCRD_TOOL

TOOL = 'ocrd-segment-extract-textline'

class ExtractTextline(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(ExtractTextline, self).__init__(*args, **kwargs)

    def process(self):
        """Extract textline images and texts from the workspace.
        
        Open and deserialize PAGE input files and their respective images,
        then iterate over the element hierarchy down to the line level.
        
        Extract an image for each textline (which depending on the workflow
        can already be deskewed, dewarped, binarized etc.), cropped to its
        minimal bounding box, and masked by the coordinate polygon outline.
        Apply ``feature_filter`` (a comma-separated list of image features,
        cf. :py:func:`ocrd.workspace.Workspace.image_from_page`) to skip
        specific features when retrieving derived images.
        If ``transparency`` is true, then also add an alpha channel which is
        fully transparent outside of the mask.
        
        Create a JSON file with:
        * the IDs of the textline and its parents,
        * the textline's text content,
        * the textline's coordinates relative to the line image,
        * the textline's absolute coordinates,
        * the textline's TextStyle (if any),
        * the textline's @production (if any),
        * the textline's @readingDirection (if any),
        * the textline's @primaryScript (if any),
        * the textline's @primaryLanguage (if any),
        * the textline's AlternativeImage/@comments (features),
        * the parent textregion's @type,
        * the page's @type,
        * the page's DPI value.
        
        Create a plain text file for the text content, too.
        
        Write all files in the directory of the output file group, named like so:
        * ID + '.raw.png': line image (if the workflow provides raw images)
        * ID + '.bin.png': line image (if the workflow provides binarized images)
        * ID + '.nrm.png': line image (if the workflow provides grayscale-normalized images)
        * ID + '.json': line metadata.
        * ID + '.gt.txt': line text.
        
        (This is intended for training and evaluation of OCR models.)
        """
        categories = [{'id': 0, 'name': 'BG'},
                    {'id': 1, 'name': 'Textline'}]

        
        # COCO Datastructure
        images = list()
        annotations = list()
        LOG = getLogger('processor.ExtractTextlines')
        assert_file_grp_cardinality(self.input_file_grp, 1)
        assert_file_grp_cardinality(self.output_file_grp, 2)
        image_file_grp, json_file_grp = self.output_file_grp.split(',')
        # pylint: disable=attribute-defined-outside-init
        LOG.info(categories[1]["id"])
        i = 0
        for n, input_file in enumerate(self.input_files):
            page_id = input_file.pageId or input_file.ID
            num_page_id = n
            LOG.info("INPUT FILE %i / %s", n, page_id)
            pcgts = page_from_file(self.workspace.download_file(input_file))
            
            self.add_metadata(pcgts)
            page = pcgts.get_Page()
            score = 1.0
            page_image, page_coords, page_image_info = self.workspace.image_from_page(
                page, page_id)
                #feature_filter=self.parameter['feature_filter'],
                #transparency=self.parameter['transparency'])
            if page_image_info.resolution != 1:
                dpi = page_image_info.resolution
                if page_image_info.resolutionUnit == 'cm':
                    dpi = round(dpi * 2.54)
            else:
                dpi = None

            regions = page.get_AllRegions(classes=['Text'], order='reading-order')
            if not regions:
                LOG.warning("Page '%s' contains no text regions", page_id)
            for region in regions:
                region_image, region_coords = self.workspace.image_from_segment(
                    region, page_image, page_coords)
                    #feature_filter=self.parameter['feature_filter'],
                    #transparency=self.parameter['transparency'])              
                lines = region.get_TextLine()
                if not lines:
                    LOG.warning("Region '%s' contains no text lines", region.id)
                for line in lines:
                    line_image, line_coords = self.workspace.image_from_segment(
                        line, region_image, region_coords)
                        #feature_filter=self.parameter['feature_filter'],
                        #transparency=self.parameter['transparency'])
                    polygon = coordinates_of_segment(
                        line, page_image, page_coords)
                    
                   
                    polygon2 = polygon.reshape(1, -1).tolist()
                    xywh = xywh_from_polygon(polygon.tolist())
                    poly = Polygon(polygon.tolist())
                    area = poly.area
                    # COCO: add annotations
                    i += 1
                    # LOG.info([xywh['x'], xywh['y'], xywh['w'], xywh['h']])
                    
                    annotations.append(
                        {'id': i, 'image_id': num_page_id,
                        'category_id': categories[1]["id"],
                        'segmentation': polygon2,
                        'area': area,
                        'bbox': [xywh['x'], xywh['y'], xywh['w'], xywh['h']],
                        'score': score,
                        'iscrowd': 0})

            file_id = make_file_id(input_file, image_file_grp)
            file_path = self.workspace.save_image_file(page_image,
                                                       file_id,
                                                       image_file_grp,
                                                       page_id=page_id,
                                                       mimetype='image/png')
            # add regions to COCO JSON for all pages
            images.append({
                # COCO does not allow string identifiers:
                # -> use numerical part of page_id
                'id': num_page_id,
                # all exported coordinates are relative to the cropped page:
                # -> use that for reference (instead of original page.imageFilename)
                'file_name': file_path,
                # -> use its size (instead of original page.imageWidth/page.imageHeight)
                'width': page_image.width,
                'height': page_image.height})

        # write COCO JSON for all pages
        file_id = json_file_grp + '.coco'
        LOG.info('Writing COCO result file "%s" in "%s"', file_id, json_file_grp)
        self.workspace.add_file(
            ID='id' + file_id,
            file_grp=json_file_grp,
            pageId=None,
            local_filename=file_id + '.json',
            mimetype='application/json',
            content=json.dumps(
                {'categories': categories,
                 'images': images,
                 'annotations': annotations},
                indent=2))

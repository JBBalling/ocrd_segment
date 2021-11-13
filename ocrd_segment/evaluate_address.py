from __future__ import absolute_import

import json, os
from shapely.geometry import Polygon
from PIL import Image, ImageDraw

from ocrd_utils import (
    getLogger,
    make_file_id,
    assert_file_grp_cardinality,
    coordinates_of_segment,
    xywh_from_polygon,
    MIMETYPE_PAGE
)
from ocrd_models.ocrd_page import TextLineType
from ocrd_modelfactory import page_from_file
from ocrd import Processor

import pathlib
from .config import OCRD_TOOL
from ocrd_cor_asv_ann.lib.alignment import Alignment, Edits
from math import fabs, sqrt, tanh
import numpy as np
from copy import deepcopy
import cv2
from matplotlib import pyplot as plt



TOOL = 'ocrd-segment-evaluate-address'

CATEGORIES = [
    'address-contact',
    'address-rcpt',
    'address-sndr'
]
IOU_THRESHOLD = 0.2
COMPARE_CLASSES = False

# Calculation based on this Paper: https://arxiv.org/pdf/2101.08418.pdf

class EvaluateAddress(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools'][TOOL]
        kwargs['version'] = OCRD_TOOL['version']
        super(EvaluateAddress, self).__init__(*args, **kwargs)


    def process(self):
        """Align textlines of multiple file groups and calculate distances.
        
        Find files in all input file groups of the workspace for the same
        pageIds. The first file group serves as reference annotation (ground truth).
        
        Open and deserialise PAGE input files, then iterate over the element
        hierarchy down to the TextLine level, looking at each first TextEquiv.
        Align character sequences in all pairs of lines for the same TextLine IDs,
        and calculate the distances using the error metric `metric`. Accumulate
        distances and sequence lengths per file group globally and per file,
        and show each fraction as a CER rate in the log.
        """

        LOG = getLogger('processor.EvaluateAdress')
        assert_file_grp_cardinality(self.output_file_grp, 1, msg="For the Evaluation of two Filegroups the result is represented in a single Filegroup")

        #metric = self.parameter['metric'] if self.parameter['metric'] is not None else 'Levenshtein-fast'
        #gtlevel = self.parameter['gt_level'] if self.parameter['metric'] is not None else None
        #confusion = self.parameter['confusion'] if self.parameter['metric'] is not None else None
        #histogram = self.parameter['histogram'] if self.parameter['metric'] is not None else None
        confusion, histogram = None, None
        metric = 'Levenshtein-fast'
        LOG.info('Using evaluation metric "%s".', metric)
        
        # compare 2 FileGroups and calculate distance of OCR-Results in between those FileGroups
        ifgs = self.input_file_grp.split(",") # input file groups
        if len(ifgs) < 2:
            raise Exception("need multiple input file groups to compare")

        # get separate aligners (1 more than needed), because
        # they are stateful (confusion counts):
        self.caligners = [Alignment(logger=LOG, confusion=bool(confusion)) for _ in ifgs]
        self.waligners = [Alignment(logger=LOG) for _ in ifgs]

        
        # running edit counts/mean/variance for each file group:
        cedits = [Edits(logger=LOG, histogram=histogram) for _ in ifgs]
        wedits = [Edits(logger=LOG) for _ in ifgs]


        # get input files:
        LOG.debug('Anzahl Input File Groups: %s', str(len(self.input_file_grp.split(","))))
        ifts = self.zip_input_files(mimetype=MIMETYPE_PAGE)

        # get input files:
        for ift in ifts:
            # running edit counts/mean/variance for each file group for this file:
            file_cedits = [Edits(logger=LOG, histogram=histogram) for _ in ifgs]
            file_wedits = [Edits(logger=LOG) for _ in ifgs]
            # get input lines:
            file_regions = [{} for _ in ifgs] # line dicts for this file
            mask_by_class = [{} for _ in ifgs]
            # input_file=specific PAGEXML
            for i, input_file in enumerate(ift):
                if not i:
                    LOG.info("processing page %s", input_file.pageId)
                if not input_file:
                    # file/page was not found in this group
                    continue
                LOG.info("INPUT FILE for %s: %s", ifgs[i], input_file.ID)
                page_id = input_file.pageId or input_file.ID
                pcgts = page_from_file(self.workspace.download_file(input_file))
                self.add_metadata(pcgts)
                file_regions[i], mask_by_class[i] = self.page_get_lines(pcgts, page_id)
                #image_dimensions[i] = page_image.size

            # compare lines_a with lines_b -> Calculate distance:
            #LOG.info(file_regions)
            LOG.info("len(file_regions[0].keys()): %s and len(file_regions[1].keys()): %s", str(len(file_regions[0].keys())), str(len(file_regions[1].keys())))
            
            # make sure that binarized_candidates_by_class_GT[i] ∩ binarized_candidates_by_class_GT[j] = ∅, für i != j
            candidates_by_class_GT = mask_by_class[0]
            candidates_by_class_PRED = mask_by_class[1]

            test_candidates_GT = {
                1: [np.array([100, 100, 500, 500] ,dtype=np.int16), np.array([100, 550, 500, 950] ,dtype=np.int16), np.array([1000, 100, 1400, 500] ,dtype=np.int16), np.array([1000, 600, 1400, 900], dtype=np.int16)]
            }
            test_candidates_PRED = {
                1: [np.array([120, 120, 500, 950] ,dtype=np.int16), np.array([1080, 180, 1200, 450] ,dtype=np.int16), np.array([1280, 180, 1350, 450] ,dtype=np.int16), np.array([1250, 600, 1400, 900], dtype=np.int16)]
            }

            img = np.zeros((1000, 2000,3), dtype=np.uint8)
            img_pred = np.zeros((1000, 2000,3), dtype=np.uint8)
            img_gt = np.zeros((1000, 2000,3), dtype=np.uint8)
            gt_color = (255, 0, 0)
            pred_color = (0, 255, 0)
            thikness = 2


            for key in test_candidates_GT.keys():
                for i in range(0, len(test_candidates_GT[key])):
                    img = cv2.rectangle(img, (test_candidates_GT[key][i][0], test_candidates_GT[key][i][1]), (test_candidates_GT[key][i][2], test_candidates_GT[key][i][3]), gt_color, thikness)
                    img_gt = cv2.rectangle(img_gt, (test_candidates_GT[key][i][0], test_candidates_GT[key][i][1]), (test_candidates_GT[key][i][2], test_candidates_GT[key][i][3]), gt_color, thikness)
                    #test_candidates_GT[key][i] = calculate_binarized_masks(img_gt, test_candidates_GT[key][i])
                    #test_candidates_PRED[key][i] = calculate_binarized_masks(img_gt, test_candidates_PRED[key][i])
            for key in test_candidates_PRED.keys():
                for i in range(0, len(test_candidates_PRED[key])):
                    img = cv2.rectangle(img, (test_candidates_PRED[key][i][0], test_candidates_PRED[key][i][1]), (test_candidates_PRED[key][i][2], test_candidates_PRED[key][i][3]), pred_color, thikness)
                    img_pred = cv2.rectangle(img_pred, (test_candidates_PRED[key][i][0], test_candidates_PRED[key][i][1]), (test_candidates_PRED[key][i][2], test_candidates_PRED[key][i][3]), pred_color, thikness)
                
                    
            cv2.imwrite('TESTIMAGE.jpg', img)
            cv2.imwrite('TESTIMAGE_GT.jpg', img_gt)
            cv2.imwrite('TESTIMAGE_PRED.jpg', img_pred)

            #test_candidates_GT = create_valid_Candidates(test_candidates_GT)
            #test_candidates_PRED = create_valid_Candidates(test_candidates_PRED)
            
            LOG.info("****************GT*************************")
            LOG.info(candidates_by_class_GT)
            LOG.info("****************PRED*************************")
            LOG.info(candidates_by_class_PRED)


            # This representation identifies a model-predicted region as under-
            # segmenting when it overlaps with at least two different ground-
            # truth regions gk ∈ GIb and gl ∈ GIb. Similarly, a ground-
            # truth region is involved in under-segmentation when it overlaps
            # with a prediction region that in turn overlaps with at least two
            # ground-truth regions. This is represented as GIU = {gi ∈GIb},
            # s.t. ∃j ∈[1..N ], ∃l ∈[1..M ], i 6= j, sl ∩gi 6= ∅∧sl ∩gj 6= ∅.
            # This representation counts ground-truth regions gi ∈GIb that
            # overlap with at least one prediction region sl ∈SIb, while the
            # prediction region sl overlaps with at least one other prediction
            # region gj ∈ GIb.
    
            # We denote regions in SIb contributing to over-segmentation
            # as SIO = {si ∈ SIb}, where si ∩ gl != ∅ ∧ sj ∩ gl != ∅, 
            # i ∈ [1, ··· , M ], j ∈ [1, ··· , M ], l ∈ [1, ··· , N ], i 6= j

            def get_regions_that_overlap_with_regions_which_itsself_overlap_with_another_region(candidates_to_count, candidates_to_check):
                result = {}
                for key, region_count in candidates_to_count.items():
                    result[key] = []
                    for key2, region_check in candidates_to_check.items():
                        if key == key2:
                            for r1 in region_count:
                                for r2 in region_check:
                                    if is_overlapping(r1, r2):
                                        # region count intersects region check -> test if region_check intersects another region_count
                                        candidates_to_further_check = candidates_to_count
                                        for key3, region_count2 in candidates_to_further_check.items():
                                            if key3 == key2:
                                                for r3 in region_count2:
                                                    #rect_3 = get_rectangles_from_binaryImage(r3)
                                                    if np.array_equal(r3, r1):
                                                        continue
                                                    if is_overlapping(r3, r2):
                                                        if np.any(np.all(r1 != result[key], axis=0)):
                                                            result[key].append(r1)

                
                return result

            # calculates regions, which overlap with at least 2 other regions (either GT -> Oversegmentation OR PRED -> Undersegmentation)
            def get_regions_overlapping_more_than_one_region(GT_candidates, PRED_candidates):
                result = {}
                for gt_key, gt_region in GT_candidates.items():
                    result[gt_key] = []
                    for pred_key, pred_region in PRED_candidates.items():
                        if gt_key == pred_key:  
                            for r1 in gt_region:
                                number_of_intersections = 0
                                for r2 in pred_region:
                                    if is_overlapping(r1, r2):
                                        number_of_intersections += 1

                                if number_of_intersections > 1:
                                    if np.any(np.all(r1 != result[gt_key], axis=0)):
                                        result[gt_key].append(r1)

                return result

          
            oversegmentations_PRED_byclass = get_regions_that_overlap_with_regions_which_itsself_overlap_with_another_region(candidates_by_class_PRED,candidates_by_class_GT)
            oversegmentations_GT_byclass = get_regions_overlapping_more_than_one_region(candidates_by_class_GT, candidates_by_class_PRED)

            undersegmentations_PRED_byclass = get_regions_overlapping_more_than_one_region(candidates_by_class_PRED, candidates_by_class_GT)
            undersegmentations_GT_byclass = get_regions_that_overlap_with_regions_which_itsself_overlap_with_another_region(candidates_by_class_GT, candidates_by_class_PRED)

            print_rum_or_rom(candidates_by_class_GT, candidates_by_class_PRED, oversegmentations_GT_byclass, oversegmentations_PRED_byclass, "ROM")
            print_rum_or_rom(candidates_by_class_GT, candidates_by_class_PRED, undersegmentations_GT_byclass, undersegmentations_PRED_byclass, "RUM")
            


            overseg_testdata_pred = get_regions_that_overlap_with_regions_which_itsself_overlap_with_another_region(test_candidates_PRED, test_candidates_GT)
            overseg_testdata_gt = get_regions_overlapping_more_than_one_region(test_candidates_GT, test_candidates_PRED)

            underseg_testdata_pred = get_regions_overlapping_more_than_one_region(test_candidates_PRED, test_candidates_GT)
            underseg_testdata_gt = get_regions_that_overlap_with_regions_which_itsself_overlap_with_another_region(test_candidates_GT, test_candidates_PRED)
            LOG.info("TESTDATEN")
            print_rum_or_rom(test_candidates_GT, test_candidates_PRED, overseg_testdata_gt, overseg_testdata_pred, "ROM")
            print_rum_or_rom(test_candidates_GT, test_candidates_PRED, underseg_testdata_gt, underseg_testdata_pred, "RUM")

            #intersecting_dict = count_intersecting_regions(test_candidates_GT, test_candidates_PRED)
            
            
            # PAARE NACH IoU und KLASSEN BILDEN -> DANN PAARE ALIGNIEREN UND EVALUIEREN
            f_regions = deepcopy(file_regions)
            max_number_of_predictions = len(file_regions[1].keys())
            pairs = []
            for i, gt_region in enumerate(file_regions[0].keys()):
                if i >= max_number_of_predictions:
                    LOG.info("Preventing KeyError: GT-List is longer than Predicted-List! Current index: %s and predictions length: %s", str(i),str(len(file_regions[1].keys())))
                    break
                best_iou = tuple([None, None]) # tuple(iou, index)
                for k, ocr_region in enumerate(list(file_regions[1].keys())):   
                    gt_region_box = array_from_tuple(gt_region)
                    ocr_region_box = array_from_tuple(ocr_region)
                    iou = IoU(gt_region_box, ocr_region_box)
                    LOG.info("The intersection over union of pair (%s, %s) is %f" % (gt_region, ocr_region, iou))
                    if iou > IOU_THRESHOLD:
                        if COMPARE_CLASSES:
                            if gt_region[4] == ocr_region[4]:
                                if best_iou[0] is not None:
                                    if iou > best_iou[0]:
                                        best_iou = ([iou, ocr_region])                            
                                    continue
                                else:
                                    best_iou = ([iou, ocr_region])
                        else:
                            if best_iou[0] is not None:
                                if iou > best_iou[0]:
                                    best_iou = ([iou, ocr_region])                            
                                continue
                            else:
                                best_iou = ([iou, ocr_region])
                if best_iou[1] is not None:
                    pairs.append(tuple([gt_region, best_iou[1]]))
                    del file_regions[1][best_iou[1]]

            LOG.info("***********************************")
            LOG.info(f_regions)
            LOG.info(pairs)
            
            report = dict()
            for n, pair in enumerate(pairs):
                for i, input_file in enumerate(ift):
                    if not i:
                        continue
                    ifg_pair = ifgs[0] + ", " + ifgs[i]
                    regions = report.setdefault(ifg_pair, dict()).setdefault('regions', list())
                    if not input_file:
                        continue
                    gt_region_id, ocr_region_id = pair[0], pair[1]
                    LOG.info("gt_region_box: %s with Textelement %s",  str(gt_region_id), str(f_regions[0][gt_region_id]))
                    LOG.info("ocr_region_box %s with Textelement %s ", str(ocr_region_id), str(f_regions[1][ocr_region_id]))
                    gt_region_box, ocr_region_box = array_from_tuple(gt_region_id), array_from_tuple(ocr_region_id)
                    # Calculate distance between a and b for each address
                    gt_regions = f_regions[0][gt_region_id]
                    len_gt = len(gt_regions)
                    lines_gt = gt_regions.split()

                    region_ocr = f_regions[1][ocr_region_id]
                    len_ocr = len(region_ocr)
                    lines_ocr = region_ocr.split()

                    if 0.2 * (len_gt + len_ocr) < fabs(len_gt - len_ocr) > 5:
                                LOG.warning('Region "%s" in file "%s" deviates significantly in length (%d vs %d)',
                                            gt_region_id, input_file.ID, len_gt, len_ocr)
                                            
                    if metric == 'Levenshtein-fast':
                        # not exact (but fast): codepoints
                        cdist = self.caligners[i].get_levenshtein_distance(region_ocr, gt_regions)
                        wdist = self.waligners[i].get_levenshtein_distance(lines_ocr, lines_gt)

                    else:
                        # exact (but slow): grapheme clusters
                        cdist = self.caligners[i].get_adjusted_distance(region_ocr, gt_regions,
                                                                        # Levenshtein / NFC / NFKC / historic_latin
                                                                        normalization=metric)
                        wdist = self.waligners[i].get_adjusted_distance(lines_ocr, lines_gt,
                                                                        # Levenshtein / NFC / NFKC / historic_latin
                                                                        normalization=metric)
                    
                    # align and accumulate edit counts for regions:
                    file_cedits[i].add(cdist, region_ocr, gt_regions)
                    file_wedits[i].add(wdist, lines_ocr, lines_gt)
                    # todo: maybe it could be useful to retrieve and store the alignments, too
                    regions.append({str(gt_region_id): {
                        'char-length': len_gt,
                        'char-error-rate': cdist,
                        'word-error-rate': wdist,
                        #'line-error-rate': ldist,
                        'gt': gt_regions,
                        'ocr': region_ocr}})
                

            # report results for file
            for i, input_file in enumerate(ift):
                if not i:
                    continue
                elif not input_file:
                    # file/page was not found in this group
                    continue
                LOG.info("%5d regions %.3f±%.3f CER %.3f±%.3f WER %s / %s vs %s",
                         file_cedits[i].length,
                         file_cedits[i].mean, sqrt(file_cedits[i].varia),
                         file_wedits[i].mean, sqrt(file_wedits[i].varia),
                         input_file.pageId, ifgs[0], ifgs[i])
                pair = ifgs[0] + ',' + ifgs[i]
                if pair in report:
                    report[pair]['num-lines'] = file_cedits[i].length
                    report[pair]['char-error-rate-mean'] = file_cedits[i].mean
                    report[pair]['char-error-rate-varia'] = file_cedits[i].varia
                    report[pair]['word-error-rate-mean'] = file_wedits[i].mean
                    report[pair]['word-error-rate-varia'] = file_wedits[i].varia
                    # accumulate edit counts for files
                    cedits[i].merge(file_cedits[i])
                    wedits[i].merge(file_wedits[i])
                else:
                    continue

            # write back result to page report
            LOG.info(ift[0])
            file_id = make_file_id(ift[0], self.output_file_grp)
            LOG.info(file_id)
            file_path = os.path.join(self.output_file_grp, file_id + '.json')
            self.workspace.add_file(
                ID=file_id,
                file_grp=self.output_file_grp,
                pageId=input_file.pageId,
                local_filename=file_path,
                mimetype='application/json',
                content=json.dumps(report, indent=2, ensure_ascii=False))

            LOG.debug("FINAL REPORT AND IFGS LENGTH %s IFTS LENGTH %s IFT LENGTH %s", str(len(ifgs)), str(len(ifts)), str(len(ift)))
            LOG.debug(report)

        # report overall results
        report = dict()
        for i in range(1, len(ifgs)):
            if not cedits[i].length:
                LOG.warning('%s had no textlines whatsoever', ifgs[i])
                continue
            LOG.info("%5d region %.3f±%.3f CER %.3f±%.3f WER overall / %s vs %s",
                    cedits[i].length,
                    cedits[i].mean, sqrt(cedits[i].varia),
                    wedits[i].mean, sqrt(wedits[i].varia),
                    ifgs[0], ifgs[i])
            report[ifgs[0] + ',' + ifgs[i]] = {
                'num-lines': cedits[i].length,
                'char-error-rate-mean': cedits[i].mean,
                'char-error-rate-varia': cedits[i].varia,
                'word-error-rate-mean': wedits[i].mean,
                'word-error-rate-varia': wedits[i].varia,
            }
        if confusion:
            for i in range(1, len(ifgs)):
                if not cedits[i].length:
                    continue
                conf = self.caligners[i].get_confusion(confusion)
                LOG.info("most frequent confusion / %s vs %s: %s",
                        ifgs[0], ifgs[i], conf)
                report[ifgs[0] + ',' + ifgs[i]]['confusion'] = repr(conf)
        if histogram:
            for i in range(1, len(ifgs)):
                if not cedits[i].length:
                    continue
                hist = cedits[i].hist()
                LOG.info("character histograms / %s vs %s: %s",
                        ifgs[0], ifgs[i], hist)
                report[ifgs[0] + ',' + ifgs[i]]['histogram'] = repr(hist)
        # write back result to overall report
        file_id = self.output_file_grp
        file_path = os.path.join(self.output_file_grp, file_id + '.json')
        self.workspace.add_file(
            ID=file_id,
            file_grp=self.output_file_grp,
            pageId=None,
            local_filename=file_path,
            mimetype='application/json',
            content=json.dumps(report, indent=2, ensure_ascii=False))
            


    def page_get_lines(self, pcgts, page_id):
        '''Get all Regions, which belongs to either 
        
        - ADDRESS_ZIP_CITY
        - ADDRESS_FULL
        - address-contact
        - address-rcpt
        - address-sndr
        - ADDRESS_STREET_HOUSENUMBER_ZIP_CITY
        
        in the page.
        
        Iterate the element hierarchy of the page `pcgts` down
        to the Region level. For each region, store the element
        ID and its first TextEquiv annotation.
        
        Return the stored dictionary.
        '''
        LOG = getLogger('processor.EvaluateLines')
        result = dict()
        regions = pcgts.get_Page().get_AllRegions(classes=['Text'], order='reading-order')
        page = pcgts.get_Page()
        page_image, page_coords, page_image_info = self.workspace.image_from_page(
            page, page_id,
            feature_filter='binarized',
            transparency=False)
        if page_image_info.resolution != 1:
            dpi = page_image_info.resolution
            if page_image_info.resolutionUnit == 'cm':
                dpi = round(dpi * 2.54)
            zoom = 300.0 / dpi
        else:
            dpi = None
            zoom = 1.0
        if zoom < 0.7:
            LOG.info("scaling %dx%d image by %.2f", page_image.width, page_image.height, zoom)
            # actual resampling: see below
            zoomed = zoom
        else:
            zoomed = 1.0
            
        if not regions:
            LOG.warning("Page contains no text regions")
        masks_by_class = dict()
        for key in range(len(CATEGORIES)):
            masks_by_class[key] = []
        for region in regions:
            address_type = get_address_type(region)
            if address_type in CATEGORIES:
                polygon = coordinates_of_segment(region, page_image, page_coords)
                if zoomed != 1.0:
                    polygon = np.round(polygon * zoomed).astype(np.int32)    
                xywh = xywh_from_polygon(polygon)
                textequivs = region.get_TextEquiv()
                result[tuple(xywh.values()) + (address_type,)] = textequivs[0].Unicode
                current_class = CATEGORIES.index(address_type)
                masks_by_class[current_class].append(list(xywh.values()))
                                
        
            # If no regions are found/classified go to TextLine-Level
            '''lines = region.get_TextLine()
            if not lines:
                LOG.warning("Region '%s' contains no text lines", region.id)
            for line in lines:
                textequivs = line.get_TextEquiv()
                if not textequivs:
                    LOG.warning("Line '%s' contains no text results", line.id)
                    continue
                address_type = get_address_type(line)
                if address_type in CATEGORIES:
                    polygon = coordinates_of_segment(region, page_image, page_coords)
                    if zoomed != 1.0:
                        polygon = np.round(polygon * zoomed).astype(np.int32)    
                    xywh = xywh_from_polygon(polygon)
                    textequivs = line.get_TextEquiv()
                    result[tuple(xywh.values()) + (address_type,)] = textequivs[0].Unicode '''
        return result, masks_by_class

def print_rum_or_rom(gt_candidates, pred_candidates, segmentation_gt, segmentation_pred, rum_or_rom):
    if rum_or_rom == "ROM":
        if len(gt_candidates) > 0 and len(pred_candidates) > 0:
            rom_by_class = {}
            if len(segmentation_gt) > 0:
                for key in segmentation_gt.keys():
                    rom_by_class[key] = 0
                    if len(segmentation_gt[key]) > 0:
                        rom_by_class[key] = ROM(segmentation_gt[key], segmentation_pred[key], gt_candidates[key], pred_candidates[key])
                    print("Der ROM-Wert der Klasse %s beträgt %s" % (CATEGORIES[key], rom_by_class[key]))
            else:
                for key in gt_candidates.keys():
                    rom_by_class[key] = 0
                    if len(segmentation_gt[key]) > 0:
                        if key in segmentation_gt.keys() and key in segmentation_pred.keys():
                            rom_by_class[key] = ROM(segmentation_gt[key], segmentation_pred[key], gt_candidates[key], pred_candidates[key])
                        elif key in segmentation_gt.keys() and key not in segmentation_pred.keys():
                            rom_by_class[key] = ROM(segmentation_gt[key], [], gt_candidates[key], pred_candidates[key])
                        elif key not in segmentation_gt.keys() and key in segmentation_pred.keys():
                            rom_by_class[key] = ROM([], segmentation_pred[key], gt_candidates[key], pred_candidates[key])
                        else:
                            rom_by_class[key] = ROM([], [], gt_candidates[key], pred_candidates[key])
                    print("Der ROM-Wert der Klasse %s beträgt %s" % (CATEGORIES[key], rom_by_class[key]))
            #rom = ROM(oversegmentations_GT_byclass, oversegmentations_pred_byclass,binarized_candidates_by_class_GT, binarized_candidates_by_class_PRED)
        else:
            print("ROM BY CLASS COULDNT BE CALCULATED BECAUSE BINARIZED_CANDIDATES BY CLASS (GT) AND/OR BINARIZED_CANDIDATES BY CLASS (PRED) are empty.")    
    else:
        if len(gt_candidates) > 0 and len(pred_candidates) > 0:
            rum_by_class = {}
            if len(segmentation_gt) > 0:
                for key in segmentation_gt.keys():
                    rum_by_class[key] = 0
                    if len(segmentation_gt[key]) > 0:
                        rum_by_class[key] = RUM(segmentation_gt[key], segmentation_pred[key], gt_candidates[key], pred_candidates[key])
                    print("Der RUM-Wert der Klasse %s beträgt %s" % (CATEGORIES[key], rum_by_class[key]))
            else:
                for key in gt_candidates.keys():
                    rum_by_class[key] = 0
                    if len(segmentation_gt[key]) > 0:
                        if key in segmentation_gt.keys() and key in segmentation_pred.keys():
                            rum_by_class[key] = RUM(segmentation_gt[key], segmentation_pred[key], gt_candidates[key], pred_candidates[key])
                        elif key in segmentation_gt.keys() and key not in segmentation_pred.keys():
                            rum_by_class[key] = RUM(segmentation_gt[key], [], gt_candidates[key], pred_candidates[key])
                        elif key not in segmentation_gt.keys() and key in segmentation_pred.keys():
                            rum_by_class[key] = RUM([], segmentation_gt[key], gt_candidates[key], pred_candidates[key])
                        else:
                            rum_by_class[key] = RUM([], [], gt_candidates[key], pred_candidates[key])
                    print("Der RUM-Wert der Klasse %s beträgt %s" % (CATEGORIES[key], rum_by_class[key]))
            #rom = ROM(oversegmentations_GT_byclass, oversegmentations_pred_byclass,binarized_candidates_by_class_GT, binarized_candidates_by_class_PRED)
        else:
            print("RUM BY CLASS COULDNT BE CALCULATED BECAUSE BINARIZED_CANDIDATES BY CLASS (GT) AND/OR BINARIZED_CANDIDATES BY CLASS (PRED) are empty.")    


    pass
            
def is_overlapping(boxA, boxB):
    val = False
    x1_A, x1_B = boxA[0], boxB[0]
    y1_A, y1_B = boxA[1], boxB[1]
    x2_A, x2_B = boxA[2], boxB[2]
    y2_A, y2_B = boxA[3], boxB[3]
    if (x1_A < x2_B and x1_B < x2_A and y1_A < y2_B and y1_B < y2_A):
        val = True
    # gt_color, pred_color = (255, 0, 0), (0, 255, 0)
    # thikness = 2
    # img = np.zeros((1000, 2000, 3), dtype=np.uint8)
    # img = cv2.rectangle(img, (boxA[0], boxA[1]), (boxA[2], boxA[3]), gt_color, thikness)
    # img = cv2.rectangle(img, (boxB[0], boxB[1]), (boxB[2], boxB[3]), pred_color, thikness)
    # imgplot = plt.imshow(img)
    # plt.show()
    return val
    
def get_address_type(segment):
    custom = segment.get_custom()
    if not custom:
        return ''
    if 'subtype: ' in custom:
        custom = custom.replace('subtype: ', '')
    elif 'subtype:' in custom:
        custom = custom.replace('subtype:', '')
    else:
        custom = ''
    print("CUSTOM: %s " % str(custom))
    return custom

def array_from_tuple(tuple):
    return [tuple[0],tuple[1],tuple[2],tuple[3]]

def IoU(boxA, boxB):
    # calc intersection rectangle
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    y2 = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    intersect = max(0, x2 - x1) * max(0, y2 - y1)
    area_boxA = boxA[2] * boxA[3]
    area_boxB = boxB[2] * boxB[3]
    iou = intersect / float(area_boxA + area_boxB - intersect)
    return iou


# ROR = (number of GT-Regions counted towards OS (-> GT_OS) multiplied with the number of PRED-Regions counted towards OS (-> PRED_OS)) divided by (number of binarized GT-Hyperplanes -> (GT_B) * number of binarized PRED-Hyperplanes -> (PRED_B))
# ROR = (#GT_OS * #PRED_OS)/(#GT_B * #PRED_B)
def ROR(GT_OS, PRED_OS, GT_B, PRED_B):
    return ((len(GT_OS) * len(PRED_OS)) / (len(GT_B) * len(PRED_B)))

# penalize ROR Score based on the total number of over-segmenting prediction region
# S_gi = {s_j ∈ S_b}, such that s_j ∩ g_i != ∅
# m0 
# m0 = SUM((gi) max(#PRED_gi-1, 0) 
def calculate_m_OS(PRED_B, GT_B):
    m0 = 0
    for gt_region in GT_B:
        intersecting_regions = []
        for PRED_region in PRED_B:
            if is_overlapping(gt_region, PRED_region):
                intersecting_regions.append(PRED_region)
                
        m0 += max(len(intersecting_regions)-1, 0)
    return m0

def ROM(GT_OS, PRED_OS, GT_B, PRED_B):
    ROM = tanh(ROR(GT_OS, PRED_OS, GT_B, PRED_B) * calculate_m_OS(PRED_B, GT_B))
    print("""ROM CALCULATION VALUES: 
    ||GT_OS|| = %s,
    ||PRED_OS|| = %s,
    ||GT_B|| = %s, 
    ||PRED_B|| = %s,
    ROR = %s,
    mo = %s,
    ROM = %s""" % (len(GT_OS),len(PRED_OS),len(GT_B),len(PRED_B), str(ROR(GT_OS, PRED_OS, GT_B, PRED_B)), str(calculate_m_OS(PRED_B, GT_B)), ROM))
    return ROM

def RUM(GT_US, PRED_US, GT_B, PRED_B):
    RUM = tanh(RUR(GT_US, PRED_US, GT_B, PRED_B) * calculate_m_US(PRED_B, GT_B))
    print("""RUM CALCULATION VALUES: 
    ||GT_US|| = %s,
    ||PRED_US|| = %s,
    ||GT_B|| = %s, 
    ||PRED_B|| = %s,
    RUR = %s,
    mo = %s,
    RUM = %s""" % (len(GT_US),len(PRED_US),len(GT_B),len(PRED_B), str(RUR(GT_US, PRED_US, GT_B, PRED_B)), str(calculate_m_US(PRED_B, GT_B)), RUM))
    return 

def RUR(GT_US, PRED_US, GT_B, PRED_B):
    return ((len(GT_US) * len(PRED_US)) / (len(GT_B) * len(PRED_B)))

def calculate_m_US(PRED_B, GT_B):
    m0 = 0
    for pred_region in PRED_B:
        intersecting_regions = []
        for gt_region in GT_B:
            if is_overlapping(pred_region, gt_region):
                intersecting_regions.append(gt_region)
        m0 += max(len(intersecting_regions)-1, 0)
    return m0



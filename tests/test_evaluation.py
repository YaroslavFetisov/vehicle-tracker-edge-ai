from __future__ import annotations

import pytest

from vehicle_tracker.evaluation import (
    Comparison,
    OcrScore,
    closest,
    edit_distance,
    fold_ambiguous,
    parse_annotation,
    score,
)


def test_an_annotation_line_yields_its_plate_text():
    assert parse_annotation("eu1.jpg\t396\t340\t203\t46\tM5XSX") == ["M5XSX"]


def test_an_image_with_two_plates_yields_both():
    content = "eu9.jpg\t10\t20\t30\t40\tAB1234\neu9.jpg\t50\t60\t30\t40\tCD5678\n"

    assert parse_annotation(content) == ["AB1234", "CD5678"]


def test_a_plate_written_with_a_space_keeps_it():
    assert parse_annotation("eu2.jpg\t1\t2\t3\t4\tAB 1234") == ["AB 1234"]


def test_blank_lines_are_skipped():
    assert parse_annotation("\neu1.jpg\t1\t2\t3\t4\tAB1234\n\n") == ["AB1234"]


def test_a_truncated_line_is_rejected():
    with pytest.raises(ValueError, match="malformed"):
        parse_annotation("eu1.jpg\t396\t340\tM5XSX")


def test_identical_strings_have_no_distance():
    assert edit_distance("AB1234", "AB1234") == 0


def test_one_wrong_character_costs_one_edit():
    assert edit_distance("AB1234", "AB1284") == 1


def test_a_missing_character_costs_one_edit():
    assert edit_distance("AB1234", "AB134") == 1


def test_an_empty_reading_costs_every_character():
    assert edit_distance("AB1234", "") == 6


def test_swapped_characters_cost_two_edits():
    assert edit_distance("AB1234", "AB2134") == 2


def test_distance_does_not_depend_on_argument_order():
    assert edit_distance("AB1234", "AB134") == edit_distance("AB134", "AB1234")


def test_a_perfect_run_scores_one():
    tally = OcrScore()
    tally.add("AB1234", "AB1234")
    tally.add("CD5678", "CD5678")

    assert tally.exact_match == 1.0
    assert tally.reading_rate == 1.0
    assert tally.character_error_rate == 0.0


def test_a_near_miss_is_not_an_exact_match_but_keeps_most_characters():
    tally = OcrScore()
    tally.add("AB1234", "AB1284")

    assert tally.exact_match == 0.0
    assert tally.character_error_rate == pytest.approx(1 / 6)


def test_an_unread_plate_counts_as_every_character_wrong():
    tally = OcrScore()
    tally.add("AB1234", None)

    assert tally.readings == 0
    assert tally.reading_rate == 0.0
    assert tally.character_error_rate == 1.0


def test_confident_nonsense_is_worse_than_reading_nothing():
    missed = OcrScore()
    missed.add("AB1234", None)
    guessed = OcrScore()
    guessed.add("AB1234", "XY9876ZZ")

    assert guessed.character_error_rate > missed.character_error_rate


def test_an_empty_run_reports_zero_instead_of_dividing_by_zero():
    empty = OcrScore()

    assert empty.exact_match == 0.0
    assert empty.reading_rate == 0.0
    assert empty.character_error_rate == 0.0


def test_characters_the_plate_font_renders_alike_fold_together():
    assert fold_ambiguous("RKO99AN") == "RK099AN"
    assert fold_ambiguous("AI1234") == "A11234"


def test_folding_leaves_an_unambiguous_reading_alone():
    assert fold_ambiguous("AB1234CD") == "AB1234CD"


def test_a_label_that_writes_zero_as_the_letter_counts_as_read_once_folded():
    comparison = Comparison("eu1.jpg", "RKO99AN", "RK099AN")

    assert not comparison.correct
    assert comparison.folded().correct


def test_folding_does_not_rescue_a_genuinely_wrong_reading():
    comparison = Comparison("eu2.jpg", "AB1234", "XY5678")

    assert not comparison.folded().correct


def test_folding_keeps_an_unread_plate_unread():
    comparison = Comparison("eu3.jpg", "AB1234", None).folded()

    assert comparison.actual is None
    assert comparison.errors == 6


def test_a_single_labelled_plate_is_the_one_scored_against():
    assert closest(["AB1234"], "AB1284") == "AB1234"


def test_the_reading_is_scored_against_the_plate_it_is_closest_to():
    assert closest(["AB1234", "CD5678"], "CD5679") == "CD5678"


def test_an_unread_image_is_scored_against_the_first_labelled_plate():
    assert closest(["AB1234", "CD5678"], None) == "AB1234"


def test_a_set_of_comparisons_is_scored_together():
    result = score(
        [
            Comparison("a.jpg", "AB1234", "AB1234"),
            Comparison("b.jpg", "CD5678", "CD5679"),
            Comparison("c.jpg", "EF9012", None),
        ]
    )

    assert result.images == 3
    assert result.exact == 1
    assert result.readings == 2
    assert result.character_error_rate == pytest.approx(7 / 18)

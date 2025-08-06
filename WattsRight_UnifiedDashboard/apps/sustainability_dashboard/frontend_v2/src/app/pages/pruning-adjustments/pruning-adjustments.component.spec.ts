import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PruningAdjustmentsComponent } from './pruning-adjustments.component';

describe('PruningAdjustmentsComponent', () => {
  let component: PruningAdjustmentsComponent;
  let fixture: ComponentFixture<PruningAdjustmentsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PruningAdjustmentsComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PruningAdjustmentsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
